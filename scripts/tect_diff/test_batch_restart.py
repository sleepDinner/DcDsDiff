"""CPU policy checks for the explicitly authorized microbatch6/global12 restart."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.tect_diff.batch_restart import (import_batch_dependencies, PARENT_RUN, CHILD_RUN,
                                            PARENT_COMMIT, PROTOCOL, REFERENCE_HASH, CALIBRATION_HASH)
from scripts.tect_diff.common import atomic_json, json_hash, read_json, sha256


class BatchRestartTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[2]
        self.temp = tempfile.TemporaryDirectory(prefix='tect-batch-policy-', dir=project/'runtime')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent, self.child = (self.root/'runs'/name for name in (PARENT_RUN, CHILD_RUN))
        self.parent.mkdir(parents=True)
        self.child.mkdir(parents=True)
        self.parent_config = read_json(project/'configs/tect_diff/full_r512_s42_refnorm_v2.json')
        self.config = copy.deepcopy(self.parent_config)
        self.config['training'].update(micro_batch=6, accumulation_steps=1, global_batch=12)
        self.config['evaluation'].update(micro_batch=8, metric_workers=4, metric_queue_limit=8)
        self.config['protocol_id'] = PROTOCOL
        parent_worker = self.parent/'source/scripts/tect_diff/worker.py'
        child_worker = self.child/'source/scripts/tect_diff/worker.py'
        helper = self.child/'source/scripts/tect_diff/evaluation_pipeline.py'
        for path in (parent_worker, child_worker, helper):
            path.parent.mkdir(parents=True, exist_ok=True)
        parent_worker.write_text('class Worker:\n def evaluate(self, model, epoch):\n  return 1\n def main_training(self):\n  return 7\n')
        child_worker.write_text('from scripts.tect_diff.evaluation_pipeline import EvaluationMetricPipeline\nclass Worker:\n def evaluate(self, model, epoch):\n  return 2\n def main_training(self):\n  return 7\n')
        helper.write_text('class EvaluationMetricPipeline:\n pass\n')
        self.old = {'commit': PARENT_COMMIT, 'config_hash': json_hash(self.parent_config),
                    'source_hashes': {'model/tect_diff/evidence.py': 'old-evidence',
                                      'scripts/tect_diff/worker.py': sha256(parent_worker)}}
        candidate = self.child/'source/model/tect_diff/evidence.py'
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b'validated evidence implementation')
        self.new = {'commit': 'new-source', 'config_hash': json_hash(self.config),
                    'source_hashes': {**self.old['source_hashes'], 'model/tect_diff/evidence.py': sha256(candidate),
                                      'scripts/tect_diff/worker.py': sha256(child_worker),
                                      'scripts/tect_diff/evaluation_pipeline.py': sha256(helper)}}
        self.state = {'controller_alive': False, 'worker_alive': False, 'status': 'INTERRUPTED', 'stage': 'MAIN'}
        self.bundle = {'manifest_hashes': {'train': 'train', 'reference': 'ref', 'calibration': 'cal'}}
        self.artifact = {'metadata': {'reference_sha256': REFERENCE_HASH, 'training_manifest_sha256': 'train',
                                     'fit_manifest_sha256': 'cal', 'protocol_id': self.parent_config['protocol_id']},
                         'artifact_hash': 'logical', 'fit_receipt': {'images': 2048}, 'buffer': torch.ones(2)}
        artifact_path = self.parent/'calibration.pth'
        torch.save(self.artifact, artifact_path)
        self.ref = {'status': 'COMPLETED', 'sha256': REFERENCE_HASH, 'epoch': 19,
                    'protocol_id': self.parent_config['protocol_id'], 'config_hash': self.old['config_hash']}
        self.cal = {'status': 'COMPLETED', 'artifact_path': str(artifact_path), 'sha256': CALIBRATION_HASH,
                    'artifact_hash': 'logical', 'fit_receipt': {'images': 2048},
                    'protocol_id': self.parent_config['protocol_id'], 'config_hash': self.old['config_hash']}
        atomic_json(self.parent/'operational_hold.json', {'active': True})
        atomic_json(self.child/'operational_hold.json', {'active': True})
        for name, value in [('data_bundle.json', self.bundle), ('reference_receipt.json', self.ref),
                            ('calibration_receipt.json', self.cal), ('reference_health.json', {}),
                            ('reference_final_health.json', {}), ('artifact_continuation.json',
                             {'parent_run': 'TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C', 'status': 'COMPLETED',
                              'config_hash': self.old['config_hash'], 'reference_sha256': self.ref['sha256'],
                              'calibration_sha256': self.cal['sha256']})]:
            atomic_json(self.parent/name, value)
        (self.parent/'reference_metrics.jsonl').write_text('{"epoch":19}\n')
        self.validation_path = self.root/'runtime/validation.json'
        self.forward_path = self.root/'runtime/tect-resource-recheck-20260915/forward/aggregate.json'
        self.forward = {'status': 'COMPLETED', 'scientific_result': False, 'global_count': 16,
                        'candidate_evidence_sha256': sha256(candidate), 'forward_original_repeat_bitwise': True,
                        'reference_calibration_rng_unchanged': True, 'all_global_ids_exact': True,
                        'variants': {'geometry_only': {'all_forward_outputs_bitwise_equal': True},
                                     'geometry_full_queries': {'all_forward_outputs_bitwise_equal': True}}}
        atomic_json(self.forward_path, self.forward)
        self.training_path = self.root/'runtime/tect-resource-recheck-20260915/micro6/aggregate.json'
        self.training_benchmark = {
            'status': 'COMPLETED', 'passed': True, 'scientific_result': False, 'mode': 'micro6',
            'parent_run': PARENT_RUN, 'parent_source_commit': PARENT_COMMIT,
            'parent_config_hash': self.old['config_hash'],
            'candidate_evidence_sha256': sha256(candidate), 'candidate_worker_sha256': sha256(child_worker),
            'reference_sha256': REFERENCE_HASH, 'calibration_sha256': CALIBRATION_HASH,
            'accepted_geometry_variant': 'geometry_full_queries',
            'full_gradient_coverage': True, 'all_rank_gradients_synchronized': True,
            'final_parameter_and_optimizer_hashes_agree': True, 'no_oom': True, 'within_memory_cap': True,
            'probe_finite': True, 'no_weights_saved': True,
            'micro_batch': 6, 'accumulation_steps': 1, 'global_batch': 12,
            'learning_rate': .0001, 'learning_rate_scaling': 'none',
            'warmup_updates': 4, 'measured_updates': 32,
            'speedup_vs_micro4_global8': 1.11, 'speedup_vs_original_micro2_global8': 1.59,
            'forward_validation': {'passed': True, 'source': str(self.forward_path), 'sha256': sha256(self.forward_path)}}
        atomic_json(self.training_path, self.training_benchmark)
        self.validation = {'passed': True, 'parent_run': PARENT_RUN, 'parent_source_commit': PARENT_COMMIT,
                           'parent_config_hash': self.old['config_hash'], 'config_hash': self.new['config_hash'],
                           'candidate_model_sha256': sha256(candidate), 'accepted_geometry_variant': 'geometry_full_queries',
                           'full_gradient_coverage': True, 'all_rank_gradients_synchronized': True,
                           'micro_batch': 6, 'accumulation_steps': 1, 'global_batch': 12,
                           'micro6_no_oom': True, 'probe_finite': True, 'measured_speedup': 1.59,
                           'measured_speedup_vs_micro4': 1.11,
                           'training_benchmark_path': str(self.training_path),
                           'training_benchmark_sha256': sha256(self.training_path),
                           'reference_sha256': REFERENCE_HASH, 'calibration_sha256': CALIBRATION_HASH,
                           'evidence_forward_validation': self.forward,
                           'evidence_forward_validation_path': str(self.forward_path),
                           'evidence_forward_validation_sha256': sha256(self.forward_path),
                           'candidate_worker_sha256': sha256(child_worker),
                           'candidate_evaluation_pipeline_sha256': sha256(helper),
                           'evaluation_validation': {'passed': True, 'finite': True, 'full_id_coverage': True,
                                                     'numerical_policy': 'BF16_BATCH_EXECUTION_V1',
                                                     'fp32_batch_control_passed': True, 'fp32_batch_control_max_abs': 2e-7,
                                                     'model_rng_unchanged': True, 'identical_array_cpu_metrics_exact': True,
                                                     'max_probability_abs_error': .001, 'binary_disagreement_count': 3,
                                                     'max_pixel_f1_abs_difference': .0002, 'max_iou_abs_difference': .0003,
                                                     'max_boundary_f1_abs_difference': .001, 'max_mae_abs_difference': .0001,
                                                     'micro_batch': 8, 'metric_workers': 4, 'metric_queue_limit': 8}}
        atomic_json(self.validation_path, self.validation)
        stack = ExitStack()
        self.addCleanup(stack.close)
        def read_run(path):
            return ((self.child, self.root, self.config, self.new) if Path(path) == self.child
                    else (self.parent, self.root, self.parent_config, self.old))
        for target, options in [
            ('scripts.tect_diff.controller.read_run', {'side_effect': read_run}),
            ('scripts.tect_diff.controller.status', {'return_value': self.state}),
            ('scripts.tect_diff.controller.verify_source', {}),
            ('scripts.tect_diff.controller.validate_receipt', {'return_value': True}),
            ('scripts.tect_diff.data.load_bundle', {'return_value': self.bundle}),
            # Production artifact hashes are fixed; the small policy fixture is
            # still actually loaded and checked, while this file identity is stubbed.
            ('scripts.tect_diff.batch_restart.sha256',
             {'side_effect': lambda path: CALIBRATION_HASH if Path(path) == artifact_path else sha256(path)}),
            ('model.tect_diff.evidence.FixedTrajectoryEvidence', {'autospec': True})]:
            stack.enter_context(patch(target, **options))

    def run_import(self):
        return import_batch_dependencies(self.child, self.parent, self.validation_path)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_import()
        self.assertTrue(read_json(self.child/'operational_hold.json')['active'])
        self.assertFalse((self.child/'batch_restart.json').exists())
        self.assertFalse((self.child/'last.pth').exists())

    def test_rebinds_only_receipt_identity_and_preserves_artifacts(self):
        original_hashes = {name: sha256(self.parent/name) for name in ('reference_receipt.json', 'calibration_receipt.json')}
        artifact_hash = sha256(self.cal['artifact_path'])
        result = self.run_import()
        for name, original in [('reference_receipt.json', self.ref), ('calibration_receipt.json', self.cal)]:
            child = read_json(self.child/name)
            self.assertEqual(child, {**original, 'protocol_id': PROTOCOL, 'config_hash': self.new['config_hash']})
            self.assertEqual(sha256(self.parent/name), original_hashes[name])
        self.assertEqual(sha256(self.cal['artifact_path']), artifact_hash)
        self.assertEqual(result['main_start_optimizer_step'], 0)
        self.assertEqual(result['main_batch'], {'micro_batch_per_rank': 6, 'accumulation_steps': 1, 'global_batch': 12})
        self.assertEqual(result['config_delta']['training.global_batch'], {'old': 8, 'new': 12})
        self.assertFalse(result['parent_main_state_reused'])
        self.assertFalse((self.child/'last.pth').exists())
        self.assertFalse((self.child/'main_probe_receipt.json').exists())
        self.assertEqual(result['inherited_artifact_continuation']['parent_run'],
                         'TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C')
        self.assertFalse(read_json(self.child/'operational_hold.json')['active'])
        self.assertTrue(read_json(self.parent/'operational_hold.json')['active'])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_import()

    def test_requires_stopped_held_parent(self):
        self.state['worker_alive'] = True
        self.assert_rejected('stopped')
        self.state['worker_alive'] = False
        atomic_json(self.parent/'operational_hold.json', {'active': False})
        self.assert_rejected('held')

    def test_rejects_unregistered_config_changes(self):
        for key, value in [('global_batch', 16), ('epochs', 99), ('learning_rate', .0002), ('micro_batch', 2)]:
            old = self.config['training'][key]
            self.config['training'][key] = value
            self.assert_rejected('configuration')
            self.config['training'][key] = old
        self.config['data']['loader_workers'] = 8
        self.assert_rejected('configuration')

    def test_rejects_main_checkpoint_or_completed_evaluation(self):
        for name in ('last.pth', 'best.pth', 'final.pth', 'epoch_summary.json', 'main_receipt.json'):
            path = self.parent/name
            path.write_bytes(b'kept parent evidence')
            self.assert_rejected('checkpoint')
            path.unlink()
        atomic_json(self.parent/'epochs/epoch-000/evaluation.json', {'evaluation_complete': True})
        self.assert_rejected('evaluation')

    def test_rejects_unregistered_source_changes(self):
        self.old['commit'] = 'wrong'
        self.assert_rejected('registered')
        self.old['commit'] = PARENT_COMMIT
        self.new['source_hashes']['model/tect_diff/reference.py'] = 'changed'
        self.assert_rejected('Unaudited source')

    def test_allows_only_the_selected_evaluation_configuration(self):
        from scripts.tect_diff.batch_restart import _validate_config, _validate_benchmark
        _validate_config(self.config, self.parent_config)
        _validate_benchmark(self.validation_path, self.config, self.new, self.root)
        for micro in (1, 2, 4, 8, 16, 32):
            for workers in (0, 4):
                if (micro, workers) == (8, 4):
                    continue
                self.config['evaluation'].update(micro_batch=micro, metric_workers=workers)
                with self.assertRaisesRegex(ValueError, 'evaluation'):
                    _validate_config(self.config, self.parent_config)

    def test_rejects_invalid_eval_config_or_failed_versioned_numerical_policy(self):
        from scripts.tect_diff.batch_restart import _validate_config
        for key, value in [('micro_batch', 3), ('metric_workers', 8), ('metric_queue_limit', 16)]:
            old = self.config['evaluation'][key]
            self.config['evaluation'][key] = value
            with self.assertRaisesRegex(ValueError, 'evaluation'):
                _validate_config(self.config, self.parent_config)
            self.config['evaluation'][key] = old
        for key, value in [('finite', False), ('full_id_coverage', False), ('numerical_policy', None),
                           ('max_probability_abs_error', .0078126), ('fp32_batch_control_passed', False),
                           ('fp32_batch_control_max_abs', 1.01e-5), ('model_rng_unchanged', False),
                           ('identical_array_cpu_metrics_exact', False), ('binary_disagreement_count', -1),
                           ('max_pixel_f1_abs_difference', -0.1), ('max_iou_abs_difference', None),
                           ('max_boundary_f1_abs_difference', -0.1), ('max_mae_abs_difference', 'missing'),
                           ('micro_batch', 4), ('metric_workers', 0)]:
            old = self.validation['evaluation_validation'][key]
            self.validation['evaluation_validation'][key] = value
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('evaluation validation')
            self.validation['evaluation_validation'][key] = old

    def test_discloses_nonzero_batch_rounding_and_preserves_exact_cpu_metric_requirement(self):
        result = self.run_import()
        disclosed = result['validation']['evaluation_validation']
        self.assertEqual(disclosed['binary_disagreement_count'], 3)
        self.assertEqual(disclosed['max_probability_abs_error'], .001)
        self.assertGreater(disclosed['max_pixel_f1_abs_difference'], 0)
        self.assertTrue(disclosed['identical_array_cpu_metrics_exact'])
        self.assertEqual(result['evaluation_numerical_policy']['probability_budget'], .0078125)
        self.assertIn('empirical', result['evaluation_numerical_policy']['budget_basis'])

    def test_accepts_budget_boundary_but_rejects_missing_disclosures(self):
        from scripts.tect_diff.batch_restart import _validate_benchmark
        self.validation['evaluation_validation']['max_probability_abs_error'] = .0078125
        atomic_json(self.validation_path, self.validation)
        _validate_benchmark(self.validation_path, self.config, self.new, self.root)
        del self.validation['evaluation_validation']['max_mae_abs_difference']
        atomic_json(self.validation_path, self.validation)
        self.assert_rejected('evaluation validation')

    def test_rejects_training_changes_inside_otherwise_allowed_worker_file(self):
        path = self.child/'source/scripts/tect_diff/worker.py'
        path.write_text(path.read_text().replace('return 7', 'return 8'))
        self.new['source_hashes']['scripts/tect_diff/worker.py'] = sha256(path)
        self.validation['candidate_worker_sha256'] = sha256(path)
        atomic_json(self.validation_path, self.validation)
        self.assert_rejected('outside evaluate')

    def test_rejects_unbound_evaluation_worker_or_helper(self):
        for key in ('candidate_worker_sha256', 'candidate_evaluation_pipeline_sha256'):
            old = self.validation[key]
            self.validation[key] = 'wrong'
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('evaluation validation')
            self.validation[key] = old

    def test_rejects_failed_or_unbound_benchmark(self):
        for field, value in [('passed', False), ('parent_config_hash', 'wrong'), ('config_hash', 'wrong'),
                             ('candidate_model_sha256', 'wrong'), ('accepted_geometry_variant', 'unknown'),
                             ('accepted_geometry_variant', 'original'), ('accepted_geometry_variant', 'geometry_only'),
                             ('reference_sha256', 'wrong'), ('calibration_sha256', 'wrong'),
                             ('full_gradient_coverage', False), ('all_rank_gradients_synchronized', False),
                             ('micro6_no_oom', False), ('probe_finite', False), ('measured_speedup', 1.04),
                             ('measured_speedup_vs_micro4', 1.049), ('measured_speedup_vs_micro4', None),
                             ('micro_batch', 4), ('global_batch', 8)]:
            old = self.validation[field]
            self.validation[field] = value
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('validation')
            self.validation[field] = old

    def test_rejects_missing_tampered_or_unbound_micro6_training_proof(self):
        original = copy.deepcopy(self.training_benchmark)
        for field, value in [('status', 'FAILED'), ('scientific_result', True), ('mode', 'micro4'),
                             ('candidate_evidence_sha256', 'wrong'), ('candidate_worker_sha256', 'wrong'),
                             ('reference_sha256', 'wrong'), ('calibration_sha256', 'wrong'),
                             ('full_gradient_coverage', False), ('all_rank_gradients_synchronized', False),
                             ('final_parameter_and_optimizer_hashes_agree', False), ('no_oom', False),
                             ('within_memory_cap', False), ('micro_batch', 4), ('global_batch', 8),
                             ('learning_rate', .00015), ('learning_rate_scaling', 'linear'),
                             ('speedup_vs_micro4_global8', 1.049), ('measured_updates', 0)]:
            proof = {**original, field: value}
            atomic_json(self.training_path, proof)
            self.validation['training_benchmark_sha256'] = sha256(self.training_path)
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('training benchmark')
        atomic_json(self.training_path, original)
        self.validation['training_benchmark_sha256'] = 'wrong'
        atomic_json(self.validation_path, self.validation)
        self.assert_rejected('training benchmark')
        self.training_path.unlink()
        self.assert_rejected('training benchmark')

    def test_requires_micro6_speedup_over_micro4_and_accepts_declared_boundary(self):
        from scripts.tect_diff.batch_restart import _validate_benchmark
        self.validation['measured_speedup_vs_micro4'] = 1.05
        self.training_benchmark['speedup_vs_micro4_global8'] = 1.05
        atomic_json(self.training_path, self.training_benchmark)
        self.validation['training_benchmark_sha256'] = sha256(self.training_path)
        atomic_json(self.validation_path, self.validation)
        _validate_benchmark(self.validation_path, self.config, self.new, self.root)
        del self.validation['measured_speedup_vs_micro4']
        atomic_json(self.validation_path, self.validation)
        self.assert_rejected('validation failed')

    def test_rejects_missing_or_tampered_forward_proof(self):
        for key, value in [('evidence_forward_validation', None), ('evidence_forward_validation_sha256', 'wrong'),
                           ('evidence_forward_validation_path', str(self.root/'runtime/missing-proof.json'))]:
            old = self.validation[key]
            self.validation[key] = value
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('validation')
            self.validation[key] = old
        atomic_json(self.validation_path, self.validation)
        self.forward['global_count'] = 15
        atomic_json(self.forward_path, self.forward)
        self.assert_rejected('validation')

    def test_rejects_forward_proof_that_does_not_prove_exact_registered_path(self):
        for key, value in [('forward_original_repeat_bitwise', False), ('reference_calibration_rng_unchanged', False),
                           ('all_global_ids_exact', False), ('global_count', 15), ('candidate_evidence_sha256', 'wrong')]:
            old = self.forward[key]
            self.forward[key] = value
            atomic_json(self.forward_path, self.forward)
            self.validation['evidence_forward_validation_sha256'] = sha256(self.forward_path)
            atomic_json(self.validation_path, self.validation)
            self.assert_rejected('validation')
            self.forward[key] = old
        self.forward['variants']['geometry_full_queries']['all_forward_outputs_bitwise_equal'] = False
        atomic_json(self.forward_path, self.forward)
        self.validation['evidence_forward_validation_sha256'] = sha256(self.forward_path)
        atomic_json(self.validation_path, self.validation)
        self.assert_rejected('validation')

    def test_rejects_substituted_fitting_artifacts(self):
        for name in ('reference_receipt.json', 'calibration_receipt.json'):
            value = read_json(self.parent/name)
            original = value['sha256']
            value['sha256'] = 'f' * 64
            atomic_json(self.parent/name, value)
            self.assert_rejected('registered artifact')
            value['sha256'] = original
            atomic_json(self.parent/name, value)

    def test_rejects_wrong_artifact_binding_before_copy(self):
        self.artifact['metadata']['reference_sha256'] = 'wrong'
        torch.save(self.artifact, self.cal['artifact_path'])
        self.assert_rejected('binding')
        self.assertFalse((self.child/'data_bundle.json').exists())

    def test_requires_both_controller_flags_and_exclusive_parent(self):
        from scripts.tect_diff.controller import launch
        with self.assertRaisesRegex(ValueError, 'requires its fitting parent'):
            launch('unused', CHILD_RUN)
        with self.assertRaisesRegex(ValueError, 'both'):
            launch('unused', CHILD_RUN, batch_parent='parent')
        with self.assertRaisesRegex(ValueError, 'both'):
            launch('unused', CHILD_RUN, batch_validation='validation')
        with self.assertRaisesRegex(ValueError, 'one registered'):
            launch('unused', CHILD_RUN, artifact_parent='other', batch_parent='parent', batch_validation='validation')


if __name__ == '__main__':
    unittest.main()
