"""Bounded controller/report policy tests; no processes, GPUs or remote writes."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Linux resource locking is exercised by server launch. Windows unit tests
# replace only the unavailable import; each operational lock call is mocked.
if os.name != 'posix':
    sys.modules['fcntl'] = types.ModuleType('fcntl')
    try:
        from scripts.tect_diff import pilot_controller as controller, pilot_report as report
    finally:
        del sys.modules['fcntl']
else:
    from scripts.tect_diff import pilot_controller as controller, pilot_report as report
from scripts.tect_diff.common import atomic_json, read_json, sha256


class PilotControllerTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[2]
        self.original_config = read_json(project / 'configs/tect_diff/pilot_casia2_gn8_r512_s42.json')
        self.temporary = tempfile.TemporaryDirectory(prefix='pilot-control-', dir=project / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / 'runs' / controller.RUN_ID
        self.run.mkdir(parents=True)
        self.config = copy.deepcopy(self.original_config)
        self.config['project_root'] = str(self.root)
        self.provenance = {'commit': 'source', 'config_hash': 'config'}
        self.values = self.run, self.root, self.config, self.provenance
        atomic_json(self.run / 'resolved_config.json', self.config)
        atomic_json(self.run / 'provenance.json', self.provenance)
        atomic_json(self.run / 'controller_status.json', {'status': 'COMPLETED', 'stage': 'COMPLETE', 'outcome': 'NO_GO'})
        self.record = {'epoch_index': 0, 'epoch_number': 1, 'total_epochs': 10,
                       'pixel_f1': {'Casiav1': .3, 'Columbia': .5}, 'average_test2': .4,
                       'epoch_complete': True, 'evaluation_complete': True}
        (self.run / 'metrics_per_epoch.jsonl').write_text(json.dumps(self.record) + '\n', encoding='utf-8')
        self.receipt = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'protocol_id': self.config['protocol_id'],
                        'source_commit': 'source', 'config_hash': 'config', 'gate': {'passed': False}}
        for key, filename in (('checkpoint', 'last.pth'), ('best', 'best.pth')):
            path = self.run / filename
            path.write_bytes(b'bounded fixture, not model weights')
            self.receipt[key] = {'path': str(path), 'sha256': sha256(path), 'epoch': 0}
        atomic_json(self.run / 'pilot_receipt.json', self.receipt)

    def test_registered_config_and_expansion_bounds(self):
        controller.validate_config(self.original_config, controller.RUN_ID)
        cases = [('training', 'epochs', 11), ('pilot', 'total_deadline_seconds', 14401),
                 ('pilot', 'total_deadline_seconds', float('nan')),
                 ('pilot', 'full_training_automatic_launch', True), ('pilot', 'train_authentic', 2048),
                 ('model', 'normalization', 'batchnorm'), ('runtime', 'automatic_training_retries', 1)]
        for section, key, value in cases:
            with self.subTest(key=key):
                config = copy.deepcopy(self.original_config)
                config[section][key] = value
                with self.assertRaises(ValueError):
                    controller.validate_config(config, controller.RUN_ID)
        with self.assertRaises(ValueError):
            controller.validate_config(self.original_config, controller.FITTING_PARENT)

    def test_unpublished_head_fails_before_resource_or_registration_actions(self):
        responses = {('branch', '--show-current'): 'feature/tect-diff', ('rev-parse', 'HEAD'): 'local',
                     ('status', '--porcelain', '--untracked-files=no'): '',
                     ('remote', 'get-url', '--push', 'origin'): self.config['runtime']['publish_url'],
                     ('ls-remote', 'origin', 'refs/heads/feature/tect-diff'): 'different refs/heads/feature/tect-diff'}
        with patch.object(controller, 'validate_config'), \
                patch.object(controller, 'git', side_effect=lambda root, *args: responses[args]), \
                patch.object(controller, 'acquire_all') as acquire, patch.object(controller, '_spawn') as spawn:
            with self.assertRaisesRegex(ValueError, 'Publish'):
                controller.launch(self.run / 'resolved_config.json', controller.RUN_ID)
            acquire.assert_not_called()
            spawn.assert_not_called()

    def test_completed_no_go_is_valid_but_corrupt_or_misbound_checkpoint_fails(self):
        self.assertEqual(controller.validate_pilot_receipt(self.run, self.provenance)['outcome'], 'NO_GO')
        for field, value in (('source_commit', 'another'), ('config_hash', 'another'), ('outcome', 'COMPLETE_ALL8')):
            with self.subTest(field=field):
                atomic_json(self.run / 'pilot_receipt.json', {**self.receipt, field: value})
                with self.assertRaises(ValueError):
                    controller.validate_pilot_receipt(self.run, self.provenance)
        atomic_json(self.run / 'pilot_receipt.json', self.receipt)
        (self.run / 'best.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            controller.validate_pilot_receipt(self.run, self.provenance)

    def test_ready_requires_completed_same_source_epoch_confirmation(self):
        receipt = {**self.receipt, 'outcome': 'READY_FOR_FULL', 'gate': {'passed': True}}
        atomic_json(self.run / 'pilot_receipt.json', receipt)
        confirmation = {'epoch': 0, 'config_hash': 'config', 'source_commit': 'source', 'gate': {'passed': True}}
        atomic_json(self.run / 'confirmation_result.json', confirmation)
        self.assertEqual(controller.validate_pilot_receipt(self.run, self.provenance)['outcome'], 'READY_FOR_FULL')
        for update in ({'epoch': 1}, {'source_commit': 'wrong'}, {'gate': {'passed': False}}):
            atomic_json(self.run / 'confirmation_result.json', {**confirmation, **update})
            with self.assertRaisesRegex(ValueError, 'confirmation'):
                controller.validate_pilot_receipt(self.run, self.provenance)

    def test_stop_refuses_a_reused_or_unattributable_pid(self):
        identity = {'pid': 123, 'start_ticks': 'old', 'cmdline': str(self.run) + ' pilot_controller.py'}
        with patch.object(controller, '_read_pilot', return_value=self.values), \
                patch.object(controller, 'status', return_value={'controller_identity': identity}), \
                patch.object(controller, 'same_process', return_value=False), patch.object(controller.os, 'kill', create=True) as kill:
            with self.assertRaisesRegex(RuntimeError, 'verified'):
                controller.stop(self.run)
            kill.assert_not_called()

    def test_failed_run_cannot_resume_or_be_resupervised(self):
        state = {'status': 'FAILED', 'controller_alive': False, 'worker_alive': False, 'elapsed_seconds': 5}
        atomic_json(self.run / 'controller_status.json', state)
        atomic_json(self.run / 'operational_hold.json', {'active': True, 'requires_repair': True, 'reason': 'nonfinite'})
        before = (self.run / 'controller_status.json').read_bytes()
        with patch.object(controller, '_read_pilot', return_value=self.values), \
                patch.object(controller, 'status', return_value=state), \
                patch.object(controller, 'acquire_file', return_value=nullcontext()) as lock, \
                patch.object(controller, '_spawn') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'versioned repair'):
                controller.resume(self.run)
            spawn.assert_not_called()
        # supervise needs the same close() contract as an actual file lock.
        with patch.object(controller, '_read_pilot', return_value=self.values), \
                patch.object(controller, 'status', return_value=state), \
                patch.object(controller, 'acquire_file') as lock, patch.object(controller, 'acquire_all') as resources:
            self.assertEqual(controller.supervise(self.run), state)
            resources.assert_not_called()
            lock.return_value.close.assert_called_once()
        self.assertEqual(before, (self.run / 'controller_status.json').read_bytes())

    def test_shutdown_failure_still_reports_and_releases_controller_lock(self):
        atomic_json(self.run / 'controller_status.json', {'status': 'REGISTERED'})
        with patch.object(controller, '_read_pilot', return_value=self.values), \
                patch.object(controller, 'acquire_file') as lock, \
                patch.object(controller.signal, 'signal'), patch.object(controller, 'process_identity', return_value=None), \
                patch.object(controller.os, 'getpgrp', return_value=123, create=True), \
                patch.object(controller.traceback, 'print_exc'), \
                patch.object(controller, 'verify_source', side_effect=RuntimeError('source changed')), \
                patch.object(controller, '_shutdown', side_effect=RuntimeError('shutdown fault')), \
                patch.object(report, 'generate', return_value={}) as generate, \
                patch.object(report, 'publish', return_value={'commit': 'published'}) as publish:
            controller.supervise(self.run)
            generate.assert_called_once_with(self.run)
            publish.assert_called_once_with(self.run)
            lock.return_value.close.assert_called_once()
        state = read_json(self.run / 'controller_status.json')
        self.assertEqual(state['status'], 'FAILED')
        self.assertIn('shutdown fault', state['cleanup_failure'])

    def test_report_keeps_two_set_identity_and_updates_only_pilot_ledger(self):
        root_ledger = self.root / 'experiment_ledger.md'
        root_ledger.write_text('# Existing\nHistoric All8 results remain.\n', encoding='utf-8')
        old_ledger = self.root / 'analysis_reports/experiment_ledger.csv'
        old_ledger.parent.mkdir()
        old_ledger.write_bytes(b'existing all8 ledger')
        with patch.object(report, 'read_run', return_value=self.values), \
                patch.object(report, 'acquire_file', side_effect=lambda *_: nullcontext()):
            first = report.generate(self.run)
            second = report.generate(self.run)
        self.assertEqual(first['outcome'], 'NO_GO')
        self.assertFalse(second['full_training_started'])
        self.assertEqual(old_ledger.read_bytes(), b'existing all8 ledger')
        self.assertEqual(root_ledger.read_text(encoding='utf-8').count(f'<!-- pilot:{controller.RUN_ID}:begin -->'), 1)
        document = (self.root / 'analysis_reports/runs' / controller.RUN_ID / 'report.md').read_text(encoding='utf-8')
        self.assertIn('Average Test2', document)
        self.assertNotIn('All8', document)
        self.assertIn('selection_protocol=test_selected', document)
        self.assertFalse(any(item.endswith('.pth') or 'source/' in item for item in report.publication_paths(controller.RUN_ID)))

    def test_complete_confirmation_metrics_are_in_published_report(self):
        confirmation = {'epoch': 0, 'source_commit': 'source', 'config_hash': 'config',
                        'gate': {'passed': True}, 'metrics': {**self.record,
                            'image_counts': {'Casiav1': 920, 'Columbia': 180}}}
        atomic_json(self.run / 'confirmation_result.json', confirmation)
        with patch.object(report, 'read_run', return_value=self.values), \
                patch.object(report, 'acquire_file', side_effect=lambda *_: nullcontext()):
            value = report.generate(self.run)
        self.assertEqual(value['confirmation_result'], confirmation)
        document = (self.root / 'analysis_reports/runs' / controller.RUN_ID / 'report.md').read_text(encoding='utf-8')
        self.assertIn('| Casiav1 | 920 | 0.300000 |', document)
        self.assertIn('| Columbia | 180 | 0.500000 |', document)

    def test_unverified_ready_receipt_cannot_override_failed_or_running_controller(self):
        receipt = {**self.receipt, 'outcome': 'READY_FOR_FULL'}
        atomic_json(self.run / 'pilot_receipt.json', receipt)
        for status, expected in (('FAILED', 'HOLD'), ('INTERRUPTED', 'HOLD'), ('RUNNING', 'PENDING')):
            with self.subTest(status=status):
                state = {'status': status, 'stage': 'PILOT', 'outcome': 'HOLD'}
                atomic_json(self.run / 'controller_status.json', state)
                with patch.object(report, 'read_run', return_value=self.values), \
                        patch.object(report, 'acquire_file', side_effect=lambda *_: nullcontext()):
                    result = report.generate(self.run)
                self.assertEqual(result['outcome'], expected)
                self.assertEqual(result['pilot_receipt']['outcome'], 'READY_FOR_FULL')
                with patch.object(controller, '_read_pilot', return_value=self.values), \
                        patch.object(controller, 'base_status', return_value=dict(state)):
                    self.assertEqual(controller.status(self.run)['pilot_outcome'], expected)

    def test_preparation_logs_boundaries_and_stops_before_worker_launch(self):
        atomic_json(self.run / 'controller_status.json', {'status': 'REGISTERED'})
        output = io.StringIO()
        data = types.ModuleType('scripts.tect_diff.pilot_data')
        def prepare(root, config, inherited, progress):
            progress({'stage': 'DATA_AUDIT', 'completed': 100})
            raise InterruptedError('stop at a preparation callback')
        data.prepare_pilot_bundle = prepare
        data.load_pilot_bundle = lambda *_: None
        with patch.object(controller, '_read_pilot', return_value=self.values), \
                patch.object(controller, 'acquire_file') as lock, patch.object(controller, 'acquire_all', return_value=[lock.return_value]), \
                patch.object(controller.signal, 'signal'), patch.object(controller, 'process_identity', return_value=None), \
                patch.object(controller.os, 'getpgrp', return_value=123, create=True), \
                patch.object(controller, 'verify_source'), patch.object(controller, 'runtime_environment', return_value={}), \
                patch.object(controller, 'import_fitting', return_value={'manifest_hashes': {}}), \
                patch.dict(sys.modules, {'scripts.tect_diff.pilot_data': data}), \
                patch.object(controller.subprocess, 'Popen') as spawn, \
                patch.object(report, 'generate', return_value={}), \
                patch.object(report, 'publish', return_value={'commit': 'published'}), redirect_stdout(output):
            controller.supervise(self.run)
            spawn.assert_not_called()
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([message['stage'] for message in messages], ['FITTING_REUSE', 'PREPARING', 'PREPARING'])
        self.assertEqual(messages[-1]['message']['completed'], 100)
        self.assertEqual(read_json(self.run / 'controller_status.json')['status'], 'INTERRUPTED')

    def test_incomplete_duplicate_or_nonfinite_epoch_display_is_rejected(self):
        path = self.run / 'metrics_per_epoch.jsonl'
        for text in (json.dumps({**self.record, 'epoch_complete': False}),
                     json.dumps(self.record) + '\n' + json.dumps(self.record),
                     json.dumps({**self.record, 'average_test2': float('nan')})):
            path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                report.read_epochs(path)


if __name__ == '__main__':
    unittest.main()
