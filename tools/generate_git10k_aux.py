"""Generate DcDsDiff auxiliary targets from GIT10K Image/Mask folders."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageMaskPair:
    stem: str
    image_path: Path
    mask_path: Path
    prefix: str


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def prefix_from_stem(stem: str) -> str:
    match = re.match(r"([A-Za-z0-9]+)", stem)
    return match.group(1) if match else "Unknown"


def collect_images(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def pair_image_mask_files(image_root: Path, mask_root: Path) -> list[ImageMaskPair]:
    image_files = collect_images(image_root)
    mask_files = collect_images(mask_root)
    common_stems = sorted(set(image_files) & set(mask_files), key=natural_key)
    return [
        ImageMaskPair(
            stem=stem,
            image_path=image_files[stem],
            mask_path=mask_files[stem],
            prefix=prefix_from_stem(stem),
        )
        for stem in common_stems
    ]


def make_detail_map(mask: np.ndarray, radius: float = 15.0, edge_kernel: int = 3) -> np.ndarray:
    """Create the paper's detail image from a binary mask.

    The detail image is bright at the tamper boundary and decays linearly for
    nearby pixels according to distance from the nearest boundary.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    binary = ((mask > 127).astype(np.uint8)) * 255
    if np.count_nonzero(binary) == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    kernel_size = max(3, int(edge_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    edge = cv2.subtract(dilated, eroded)
    if np.count_nonzero(edge) == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    distance = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    detail = np.clip(1.0 - distance / float(radius), 0.0, 1.0)
    return np.rint(detail * 255.0).astype(np.uint8)


def make_high_frequency_view(image: np.ndarray, cutoff_ratio: float = 0.5, boost: float = 10.0) -> np.ndarray:
    """Create the HFVG high-frequency view from an RGB image."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    arr = image.astype(np.float32) / 255.0
    height, width, channels = arr.shape
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = max(1.0, float(cutoff_ratio) * min(height, width) / 2.0)
    high_pass = ((y - cy) ** 2 + (x - cx) ** 2) >= radius**2

    output = np.empty_like(arr, dtype=np.float32)
    for channel in range(channels):
        spectrum = np.fft.fftshift(np.fft.fft2(arr[:, :, channel]))
        filtered = spectrum * high_pass
        restored = np.fft.ifft2(np.fft.ifftshift(filtered)).real
        output[:, :, channel] = np.abs(restored) * float(boost)

    output = np.clip(output, 0.0, 1.0)
    return np.rint(output * 255.0).astype(np.uint8)


def save_rgb_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_l_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def copy_image_as_png(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert(mode).save(destination)


def split_project_pairs(pairs: list[ImageMaskPair], train_ratio: float) -> dict[str, str]:
    groups: dict[str, list[ImageMaskPair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.prefix].append(pair)

    assignment: dict[str, str] = {}
    for prefix_pairs in groups.values():
        prefix_pairs.sort(key=lambda pair: natural_key(pair.stem))
        split_at = int(len(prefix_pairs) * train_ratio)
        if len(prefix_pairs) > 1:
            split_at = min(max(split_at, 1), len(prefix_pairs) - 1)
        for index, pair in enumerate(prefix_pairs):
            assignment[pair.stem] = "train" if index < split_at else "test"
    return assignment


def output_dirs(pair: ImageMaskPair, out_root: Path, layout: str, assignment: dict[str, str]) -> dict[str, Path]:
    if layout == "aux":
        return {"d": out_root / "d", "t": out_root / "t"}
    if layout == "flat":
        return {name: out_root / name for name in ("f", "m", "d", "t")}
    if layout == "project":
        split = assignment[pair.stem]
        if split == "train":
            base = out_root / "train" / "Diff"
        else:
            base = out_root / "Test" / "Diff" / pair.prefix
        return {name: base / name for name in ("f", "m", "d", "t")}
    raise ValueError(f"Unsupported layout: {layout}")


def mix_output_dirs(out_root: Path, mix_name: str) -> dict[str, Path]:
    base = out_root / "Test" / "Diff" / mix_name
    return {name: base / name for name in ("f", "m", "d", "t")}


def process_pair(
    pair: ImageMaskPair,
    dirs: dict[str, Path],
    *,
    detail_radius: float,
    edge_kernel: int,
    cutoff_ratio: float,
    boost: float,
    overwrite: bool,
    copy_inputs: bool,
) -> None:
    filename = f"{pair.stem}.png"

    if copy_inputs and "f" in dirs:
        copy_image_as_png(pair.image_path, dirs["f"] / filename, "RGB", overwrite)
    if copy_inputs and "m" in dirs:
        copy_image_as_png(pair.mask_path, dirs["m"] / filename, "L", overwrite)

    detail_path = dirs["d"] / filename
    trace_path = dirs["t"] / filename

    if overwrite or not detail_path.exists():
        with Image.open(pair.mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))
        detail = make_detail_map(mask, radius=detail_radius, edge_kernel=edge_kernel)
        save_l_png(detail_path, detail)

    if overwrite or not trace_path.exists():
        with Image.open(pair.image_path) as rgb_image:
            image = np.asarray(rgb_image.convert("RGB"))
        trace = make_high_frequency_view(image, cutoff_ratio=cutoff_ratio, boost=boost)
        save_rgb_png(trace_path, trace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True, help="Folder containing GIT10K Image files.")
    parser.add_argument("--mask-root", type=Path, required=True, help="Folder containing GIT10K Mask files.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root folder.")
    parser.add_argument(
        "--layout",
        choices=("aux", "flat", "project"),
        default="flat",
        help="aux writes d/t only; flat writes f/m/d/t; project writes train/Diff and Test/Diff/<prefix>.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Project layout train split ratio per prefix.")
    parser.add_argument("--detail-radius", type=float, default=15.0, help="Detail-map decay radius in pixels.")
    parser.add_argument("--edge-kernel", type=int, default=3, help="Odd morphological kernel size for mask boundary.")
    parser.add_argument("--cutoff-ratio", type=float, default=0.5, help="HFVG high-pass radius ratio, matching paper setting.")
    parser.add_argument("--boost", type=float, default=10.0, help="HFVG enhancement multiplier, matching paper setting.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pairs for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    parser.add_argument("--skip-input-copy", action="store_true", help="Only generate d/t when layout includes f/m.")
    parser.add_argument(
        "--with-test-mix",
        action="store_true",
        help="With project layout, also write all held-out samples to Test/Diff/<mix-name> for train.py.",
    )
    parser.add_argument("--mix-name", default="Mix", help="Name of the mixed test folder used with --with-test-mix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pair_image_mask_files(args.image_root, args.mask_root)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No matched Image/Mask pairs found.")

    assignment = split_project_pairs(pairs, args.train_ratio) if args.layout == "project" else {}
    copy_inputs = not args.skip_input_copy

    for index, pair in enumerate(pairs, 1):
        dirs = output_dirs(pair, args.out_root, args.layout, assignment)
        process_pair(
            pair,
            dirs,
            detail_radius=args.detail_radius,
            edge_kernel=args.edge_kernel,
            cutoff_ratio=args.cutoff_ratio,
            boost=args.boost,
            overwrite=args.overwrite,
            copy_inputs=copy_inputs,
        )
        if args.layout == "project" and args.with_test_mix and assignment[pair.stem] == "test":
            process_pair(
                pair,
                mix_output_dirs(args.out_root, args.mix_name),
                detail_radius=args.detail_radius,
                edge_kernel=args.edge_kernel,
                cutoff_ratio=args.cutoff_ratio,
                boost=args.boost,
                overwrite=args.overwrite,
                copy_inputs=copy_inputs,
            )
        if index == 1 or index % 500 == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] processed {pair.stem}")


if __name__ == "__main__":
    main()
