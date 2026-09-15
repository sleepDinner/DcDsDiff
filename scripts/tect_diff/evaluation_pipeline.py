"""Bounded, ordered CPU metrics overlap; inference and array ownership stay outside."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.tect_diff.metrics import per_image_metrics


def _metric_row(sample_id, probability, gt, valid, threshold, boundary_ratio):
    return {'id': sample_id, **per_image_metrics(probability, gt, valid, threshold, boundary_ratio)}


class EvaluationMetricPipeline:
    """Keep at most queue_limit unfinished/uncollected per-image metric jobs.

    The caller passes independent CPU NumPy arrays and must not mutate them
    until finish(). Workers never receive tensors or execute GPU operations.
    Zero workers preserves the original synchronous per-image metric call.
    """
    def __init__(self, workers=0, queue_limit=8):
        if type(workers) is not int or workers < 0:
            raise ValueError('Metric workers must be a nonnegative integer')
        if type(queue_limit) is not int or queue_limit < 1:
            raise ValueError('Metric queue limit must be a positive integer')
        self.queue_limit = queue_limit
        self.rows = []
        self.submitted = 0
        self.closed = False
        self._pending = deque()
        # ThreadPoolExecutor starts its threads on submit, after the caller's
        # DataLoader iterator has started its worker processes.
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='tect-metrics') if workers else None

    @property
    def pending_count(self):
        return len(self._pending)

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Metric pipeline is closed')
        return self

    def _collect_one(self):
        row = self._pending[0].result()  # Propagate the original metric exception.
        self._pending.popleft()
        self.rows.append(row)

    def submit(self, sample_id, probability, gt, valid=None, threshold=.5, boundary_ratio=.005):
        if self.closed:
            raise RuntimeError('Metric pipeline is closed')
        if not isinstance(probability, np.ndarray) or not isinstance(gt, np.ndarray) or (
                valid is not None and not isinstance(valid, np.ndarray)):
            raise TypeError('CPU metrics require NumPy probability, GT and optional valid arrays')
        if self._executor is None:
            self.rows.append(_metric_row(sample_id, probability, gt, valid, threshold, boundary_ratio))
            self.submitted += 1
            return
        while len(self._pending) >= self.queue_limit:
            self._collect_one()
        self._pending.append(self._executor.submit(
            _metric_row, sample_id, probability, gt, valid, threshold, boundary_ratio))
        self.submitted += 1
        while self._pending and self._pending[0].done():
            self._collect_one()

    def finish(self):
        while self._pending:
            self._collect_one()
        return self.rows

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.finish()
        finally:
            self.closed = True
            for future in self._pending:
                future.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._pending.clear()
        return False
