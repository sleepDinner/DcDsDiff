"""Select complete named benchmarks while preserving the original data transforms."""
from pathlib import Path

from dataset.data_val import test_dataset


class BenchmarkSubset(test_dataset):
    def __init__(self, *, datasets, **kwargs):
        super().__init__(**kwargs)
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError('Select distinct, nonempty benchmark names.')
        present = {Path(path).stem.split('__', 1)[0] for path in self.images}
        if not set(datasets) <= present:
            raise ValueError(f'Unknown benchmark names: {set(datasets) - present}')
        indices = [i for i, path in enumerate(self.images)
                   if Path(path).stem.split('__', 1)[0] in datasets]
        for key in ('images', 'gts', 'des', 'traces'):
            paths = getattr(self, key)
            setattr(self, key, [paths[i] for i in indices])
        self.size = len(indices)
