"""GIT10K data loading for the fixed paper-aligned-v1 protocol.

Images, masks, detail targets and HFVG traces are paired by exact filename stem.
Training uses synchronized horizontal flips and preserves soft mask values.
"""

from pathlib import Path
import random

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _files_by_stem(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate sample stem {path.stem!r}: {files[path.stem]} and {path}")
        files[path.stem] = path
    if not files:
        raise ValueError(f"No supported images in {root}")
    return files


def _paired_paths(image_root, gt_root, de_root, trace_root):
    roots = {"image": image_root, "gt": gt_root, "de": de_root, "trace": trace_root}
    files = {name: _files_by_stem(root) for name, root in roots.items()}
    image_stems = set(files["image"])
    for name, mapping in files.items():
        missing = sorted(image_stems - set(mapping))
        extra = sorted(set(mapping) - image_stems)
        if missing or extra:
            raise ValueError(
                f"Unpaired {name} samples in {roots[name]}: "
                f"missing={missing[:8]} ({len(missing)}), extra={extra[:8]} ({len(extra)})"
            )
    stems = sorted(image_stems)
    return [[str(files[name][stem]) for stem in stems] for name in roots]


def _read_image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def _rgb_transform(size, mean=None, std=None):
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _unit_transform(size):
    # The upstream PIL bilinear resize retains soft supervision at mask boundaries.
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


class PolypObjDataset(data.Dataset):
    """Training interface retained for existing config import paths."""

    def __init__(self, image_root, gt_root, de_root, trace_root, trainsize,
                 mean=None, std=None, horizontal_flip=True, randomPeper=False,
                 boundary_modification=False, boundary_args=None):
        if randomPeper or boundary_modification or boundary_args:
            raise ValueError("paper-aligned-v1 permits horizontal flipping only; disable legacy augmentation")
        self.trainsize = trainsize
        self.horizontal_flip = horizontal_flip
        self.images, self.gts, self.des, self.traces = _paired_paths(
            image_root, gt_root, de_root, trace_root
        )
        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = _unit_transform(trainsize)
        self.de_transform = _unit_transform(trainsize)
        # CondGaussianDiffusion.forward maps this [0, 1] tensor to [-1, 1].
        self.trace_transform = _unit_transform(trainsize)
        self.size = len(self.images)

    def get_transform(self, mean=None, std=None):
        return _rgb_transform(self.trainsize, mean, std)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        de = self.binary_loader(self.des[index])
        trace = self.rgb_loader(self.traces[index])
        if self.horizontal_flip and random.random() < 0.5:
            image, gt, de, trace = (
                item.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for item in (image, gt, de, trace)
            )
        return {
            "image": self.img_transform(image),
            "gt": self.gt_transform(gt),
            "de": self.de_transform(de),
            "trace": self.trace_transform(trace),
        }

    @staticmethod
    def rgb_loader(path):
        return _read_image(path, "RGB")

    @staticmethod
    def binary_loader(path):
        return _read_image(path, "L")

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, de_root, trace_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, de_root, trace_root, trainsize)
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)


class test_dataset(data.Dataset):
    """Evaluation returns original-resolution masks and normalized model inputs."""

    def __init__(self, image_root, gt_root, de_root, trace_root, testsize, mean=None, std=None):
        super().__init__()
        self.testsize = testsize
        self.images, self.gts, self.des, self.traces = _paired_paths(
            image_root, gt_root, de_root, trace_root
        )
        self.transform = self.get_transform(mean, std)
        # sample() bypasses training forward(), so perform its trace mapping here.
        self.trace_transform = transforms.Compose([
            _unit_transform(testsize),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.size = len(self.images)
        self.index = 0

    def get_transform(self, mean=None, std=None):
        return _rgb_transform(self.testsize, mean, std)

    @staticmethod
    def rgb_loader(path):
        return _read_image(path, "RGB")

    @staticmethod
    def binary_loader(path):
        return _read_image(path, "L")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        image = self.transform(self.rgb_loader(self.images[item]))
        trace = self.trace_transform(self.rgb_loader(self.traces[item]))
        return {
            "image": image.unsqueeze(0),
            "gt": self.binary_loader(self.gts[item]),
            "de": self.binary_loader(self.des[item]),
            "trace": trace.unsqueeze(0),
            "name": Path(self.images[item]).stem + ".png",
            "image_for_post": image,
        }

    def load_data(self):
        sample = self[self.index]
        self.index = (self.index + 1) % self.size
        return tuple(sample[key] for key in ("image", "gt", "de", "trace", "name", "image_for_post"))

    def __iter__(self):
        for item in range(self.size):
            yield self[item]
