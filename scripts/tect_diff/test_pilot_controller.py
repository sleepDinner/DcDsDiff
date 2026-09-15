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
        self.data2_config = read_json(project / 'configs/tect_diff/pilot_casia2_gn8_data2_r512_s42.json')
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

    def test_data2_run_protocol_pairing_and_exact_quarantine_count(self):
        amended = copy.deepcopy(self.original_config)
        amended['protocol_id'] = controller.RUN_PROTOCOLS[controller.RUN_ID_DATA2]
        amended['data']['quarantine_pairs'] = [
            {'id': f'CASIA2:Tp_fixture_{index}', 'image_sha256': str(index) * 64,
             'mask_sha256': 'a' * 64, 'reason': 'Unresolved LA annotation semantics'}
            for index in (1, 2)]
        controller.validate_config(self.original_config, controller.RUN_ID)
        controller.validate_config(amended, controller.RUN_ID_DATA2)
        for config, run_id in ((self.original_config, controller.RUN_ID_DATA2), (amended, controller.RUN_ID)):
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, 'protocol'):
                controller.validate_config(config, run_id)
        for pairs in (None, [], amended['data']['quarantine_pairs'][:1],
                      amended['data']['quarantine_pairs'] * 2, {}):
            invalid = copy.deepcopy(amended)
            invalid['data']['quarantine_pairs'] = pairs
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, 'exactly two'):
                controller.validate_config(invalid, controller.RUN_ID_DATA2)
        missing = copy.deepcopy(amended)
        del missing['data']['quarantine_pairs']
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            controller.validate_config(missing, controller.RUN_ID_DATA2)
        for pairs in ([], amended['data']['quarantine_pairs']):
            changed_a = copy.deepcopy(self.original_config)
            changed_a['data']['quarantine_pairs'] = pairs
            with self.assertRaisesRegex(ValueError, 'Original pilot A'):
                controller.validate_config(changed_a, controller.RUN_ID)

    def test_historical_a_status_is_readable_and_remains_held(self):
        held = {'status': 'FAILED', 'stage': 'PREPARING', 'outcome': 'HOLD',
                'controller_alive': False, 'worker_alive': False}
        hold = {'active': True, 'requires_repair': True, 'reason': 'Historical LA annotation ambiguity'}
        atomic_json(self.run / 'operational_hold.json', hold)
        with patch.object(controller, 'read_run', return_value=(self.run, self.root, self.original_config, self.provenance)), \
                patch.object(controller, 'base_status', return_value=held):
            current = controller.status(self.run)
        self.assertEqual(current['status'], 'FAILED')
        self.assertEqual(current['pilot_outcome'], 'HOLD')
        self.assertEqual(read_json(self.run / 'operational_hold.json'), hold)

    def test_n8192_accepts_only_registered_c_without_mutating_config(self):
        config = copy.deepcopy(self.data2_config)
        config['protocol_id'] = 'TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-V1'
        config['pilot'].update(train_authentic=4096, train_tampered=4096)
        before = copy.deepcopy(config)
        controller.validate_config(config, 'TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C')
        self.assertEqual(config, before)
        for run_id in (controller.RUN_ID, controller.RUN_ID_DATA2):
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, 'protocol'):
                controller.validate_config(config, run_id)

    def test_n8192_preserves_all_other_b_settings(self):
        registered = copy.deepcopy(self.data2_config)
        registered['protocol_id'] = 'TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-V1'
        registered['pilot'].update(train_authentic=4096, train_tampered=4096)
        changes = [
            (('training', 'learning_rate'), 2e-4),
            (('training', 'epochs'), 9),
            (('training', 'scheduler_epochs'), 10),
            (('training', 'lambda_image'), .5),
            (('pilot', 'total_deadline_seconds'), 7200),
            (('pilot', 'gate', 'macro_f1_min'), .4),
            (('pilot', 'gate', 'consecutive_epochs'), 2),
            (('pilot', 'quick_test_per_dataset'), 64),
            (('pilot', 'train_authentic'), 1024),
            (('pilot', 'train_tampered'), 8192),
            (('evaluation', 'suite'), 'ALL8'),
            (('evaluation', 'selection_scope'), 'different tests'),
            (('evaluation', 'threshold'), .4),
            (('sampling', 'steps'), 20),
            (('reference', 'epochs'), 1),
            (('evidence', 'gamma_max'), 1.),
            (('model', 'self_condition_probability'), 0.),
            (('data', 'loader_workers'), 8),
            (('data', 'quarantine_pairs'), []),
            (('unregistered_option',), True),
        ]
        for keys, value in changes:
            invalid = copy.deepcopy(registered)
            target = invalid
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(field='.'.join(keys)), self.assertRaises(ValueError):
                controller.validate_config(invalid, 'TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C')

    def test_n8192_cannot_redefine_the_canonical_b_anchor(self):
        config = copy.deepcopy(self.data2_config)
        config['protocol_id'] = 'TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-V1'
        config['pilot'].update(train_authentic=4096, train_tampered=4096)
        changed_anchor = copy.deepcopy(self.data2_config)
        changed_anchor['training']['learning_rate'] = config['training']['learning_rate'] = 2e-4
        with patch.object(controller, 'read_json', return_value=changed_anchor), \
                self.assertRaisesRegex(ValueError, 'canonical B'):
            controller.validate_config(config, 'TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C')

    def test_registered_a_and_b_keep_their_original_sample_budgets(self):
        for original, run_id in ((self.original_config, controller.RUN_ID), (self.data2_config, controller.RUN_ID_DATA2)):
            controller.validate_config(original, run_id)
            for field in ('train_authentic', 'train_tampered'):
                config = copy.deepcopy(original)
                config['pilot'][field] = 4096
                with self.subTest(run_id=run_id, field=field), self.assertRaisesRegex(ValueError, 'budget'):
                    controller.validate_config(config, run_id)

    def test_data_expansion_parent_binding_and_no_go_evidence(self):
        parent = self.root / 'runs' / controller.RUN_ID_DATA2
        parent.mkdir()
        run = self.root / 'runs' / controller.RUN_ID_N8192
        run.mkdir()
        provenance = {'commit': controller.DATA2_SOURCE_COMMIT, 'config_hash': controller.DATA2_CONFIG_HASH}
        state = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'controller_alive': False, 'worker_alive': False}
        terminal = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'protocol_id': self.data2_config['protocol_id'],
                    'source_commit': controller.DATA2_SOURCE_COMMIT, 'config_hash': controller.DATA2_CONFIG_HASH}
        atomic_json(parent / 'pilot_receipt.json', terminal)
        atomic_json(parent / 'data_bundle.json', {'fixture': 'frozen parent metadata'})
        terminal_hash, bundle_hash = sha256(parent / 'pilot_receipt.json'), sha256(parent / 'data_bundle.json')
        proof = {'status': 'PASSED', 'parent_train_count': 2048, 'train_count': 8192,
                 'retained_parent_train_count': 2048}
        data = types.ModuleType('scripts.tect_diff.pilot_data')
        data.validate_pilot_expansion = lambda bundle, summary: dict(proof)
        with patch.object(controller, '_read_pilot', return_value=(parent, self.root, self.data2_config, provenance)), \
                patch.object(controller, 'base_status', return_value=state), \
                patch.object(controller, 'verify_source'), \
                patch.object(controller, 'acquire_file', return_value=nullcontext()), \
                patch.object(controller, 'DATA2_TERMINAL_HASH', terminal_hash), \
                patch.object(controller, 'DATA2_BUNDLE_HASH', bundle_hash), \
                patch.dict(sys.modules, {'scripts.tect_diff.pilot_data': data}):
            result = controller.validate_data_expansion(run, self.root, {})
            self.assertEqual(read_json(run / 'data_expansion.json'), result)
            self.assertEqual(result['parent_source_commit'], controller.DATA2_SOURCE_COMMIT)
            self.assertEqual(result['parent_data_bundle_sha256'], bundle_hash)
            self.assertEqual(result['parent_terminal_sha256'], terminal_hash)
            self.assertFalse(result['parent_weights_loaded'])
            before = (run / 'data_expansion.json').read_bytes()
            cases = [(state, 'controller_alive', True), (state, 'worker_alive', True),
                     (state, 'status', 'RUNNING'), (state, 'outcome', 'READY_FOR_FULL'),
                     (provenance, 'commit', 'different'), (provenance, 'config_hash', 'different'),
                     (proof, 'parent_train_count', 1024), (proof, 'train_count', 4096),
                     (proof, 'retained_parent_train_count', 2047)]
            for target, key, invalid in cases:
                old = target[key]
                target[key] = invalid
                try:
                    with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                        controller.validate_data_expansion(run, self.root, {})
                    self.assertEqual((run / 'data_expansion.json').read_bytes(), before)
                finally:
                    target[key] = old
            for filename in ('data_bundle.json', 'pilot_receipt.json'):
                original = (parent / filename).read_bytes()
                try:
                    (parent / filename).write_bytes(original + b' ')
                    with self.subTest(filename=filename), self.assertRaises(ValueError):
                        controller.validate_data_expansion(run, self.root, {})
                    self.assertEqual((run / 'data_expansion.json').read_bytes(), before)
                finally:
                    (parent / filename).write_bytes(original)

    def test_data_expansion_rejects_parent_change_during_validation(self):
        parent = self.root / 'runs' / controller.RUN_ID_DATA2
        parent.mkdir()
        run = self.root / 'runs' / controller.RUN_ID_N8192
        run.mkdir()
        provenance = {'commit': controller.DATA2_SOURCE_COMMIT, 'config_hash': controller.DATA2_CONFIG_HASH}
        state = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'controller_alive': False, 'worker_alive': False}
        terminal = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'protocol_id': self.data2_config['protocol_id'],
                    'source_commit': controller.DATA2_SOURCE_COMMIT, 'config_hash': controller.DATA2_CONFIG_HASH}
        atomic_json(parent / 'pilot_receipt.json', terminal)
        atomic_json(parent / 'data_bundle.json', {'fixture': 'frozen parent metadata'})
        data = types.ModuleType('scripts.tect_diff.pilot_data')
        def changed(bundle, summary):
            atomic_json(parent / 'data_bundle.json', {'fixture': 'changed during validation'})
            return {'status': 'PASSED', 'parent_train_count': 2048, 'train_count': 8192,
                    'retained_parent_train_count': 2048}
        data.validate_pilot_expansion = changed
        with patch.object(controller, '_read_pilot', return_value=(parent, self.root, self.data2_config, provenance)), \
                patch.object(controller, 'base_status', return_value=state), patch.object(controller, 'verify_source'), \
                patch.object(controller, 'acquire_file', return_value=nullcontext()), \
                patch.object(controller, 'DATA2_TERMINAL_HASH', sha256(parent / 'pilot_receipt.json')), \
                patch.object(controller, 'DATA2_BUNDLE_HASH', sha256(parent / 'data_bundle.json')), \
                patch.dict(sys.modules, {'scripts.tect_diff.pilot_data': data}), \
                self.assertRaisesRegex(ValueError, 'changed during validation'):
            controller.validate_data_expansion(run, self.root, {})
        self.assertFalse((run / 'data_expansion.json').exists())

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

    def test_c_expansion_failure_blocks_fresh_and_reused_bundle_before_worker(self):
        run = self.root / 'runs' / 'TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C'
        run.mkdir()
        bundle = {'train': ['train'], 'tests': {'Casiav1': ['test'], 'Columbia': ['test']},
                  'manifest_hashes': {'reference': 'ref', 'calibration': 'cal'},
                  'summary': {}, 'summary_path': 'fixture'}
        for reused in (False, True):
            with self.subTest(reused=reused):
                atomic_json(run / 'controller_status.json', {'status': 'REGISTERED'})
                atomic_json(run / 'operational_hold.json', {'active': False})
                if reused:
                    atomic_json(run / 'data_bundle.json', {'existing': True})
                output = io.StringIO()
                data = types.ModuleType('scripts.tect_diff.pilot_data')
                data.prepare_pilot_bundle = lambda *args, **kwargs: bundle
                data.load_pilot_bundle = lambda *_: bundle
                with patch.object(controller, '_read_pilot', return_value=(run, self.root, self.config, self.provenance)), \
                        patch.object(controller, 'acquire_file') as lock, \
                        patch.object(controller, 'acquire_all', return_value=[lock.return_value]), \
                        patch.object(controller.signal, 'signal'), patch.object(controller, 'process_identity', return_value=None), \
                        patch.object(controller.os, 'getpgrp', return_value=123, create=True), \
                        patch.object(controller, 'verify_source'), patch.object(controller, 'runtime_environment', return_value={}), \
                        patch.object(controller, 'import_fitting', return_value={'manifest_hashes': bundle['manifest_hashes']}), \
                        patch.dict(sys.modules, {'scripts.tect_diff.pilot_data': data}), \
                        patch.object(controller, 'validate_data_expansion', create=True, side_effect=ValueError('expansion proof failed')) as validate, \
                        patch.object(controller.subprocess, 'Popen', side_effect=AssertionError('Unexpected worker launch')) as spawn, \
                        patch.object(controller.traceback, 'print_exc'), \
                        patch.object(report, 'generate', return_value={}), \
                        patch.object(report, 'publish', return_value={'commit': 'published'}), redirect_stdout(output):
                    controller.supervise(run)
                    validate.assert_called_once_with(run, self.root, bundle)
                    spawn.assert_not_called()
                self.assertNotIn('PREPARATION_COMPLETE', output.getvalue())
                self.assertIn('expansion proof failed', read_json(run / 'controller_status.json')['failure_reason'])
                self.assertTrue(read_json(run / 'operational_hold.json')['requires_repair'])

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
