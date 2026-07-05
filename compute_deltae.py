#!/usr/bin/env python3
"""Compute Delta E between paired GT and output images in a directory.

Example:
    python compute_deltae.py
    python compute_deltae.py --path /path/to/visualization/90000
    python compute_deltae.py --metric cie76 --save-csv deltae_results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_cie76, deltaE_ciede2000, rgb2lab


DEFAULT_PATH = (
    "/data0/home/loujunyu2/AILIVE/checkpoint/experiments/oppo_muon_datsr/visualization/60000"
)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PAIR_PATTERN = re.compile(r"^(?P<key>.+)_(?P<role>gt|output)$")


@dataclass
class PairResult:
    key: str
    gt_path: Path
    output_path: Path
    mean_delta_e: float
    median_delta_e: float
    min_delta_e: float
    max_delta_e: float
    pixel_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Delta E for paired *_gt and *_output images."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(DEFAULT_PATH),
        help=f"Directory containing paired images. Default: {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--metric",
        choices=("cie2000", "cie76"),
        default="cie2000",
        help="Delta E metric to use. Default: cie2000",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save per-image results as CSV.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many worst samples to print. Default: 10",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-image progress printing.",
    )
    return parser.parse_args()


def collect_pairs(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    gt_map: dict[str, Path] = {}
    output_map: dict[str, Path] = {}

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        match = PAIR_PATTERN.match(path.stem)
        if match is None:
            continue

        key = match.group("key")
        role = match.group("role")
        target_map = gt_map if role == "gt" else output_map

        if key in target_map:
            raise ValueError(f"Duplicate {role} image for key '{key}': {path}")
        target_map[key] = path

    return gt_map, output_map


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_delta_e_map(
    output_rgb: np.ndarray, gt_rgb: np.ndarray, metric: str
) -> np.ndarray:
    output_lab = rgb2lab(output_rgb)
    gt_lab = rgb2lab(gt_rgb)

    if metric == "cie2000":
        return deltaE_ciede2000(output_lab, gt_lab)
    if metric == "cie76":
        return deltaE_cie76(output_lab, gt_lab)
    raise ValueError(f"Unsupported metric: {metric}")


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def evaluate_pairs(
    gt_map: dict[str, Path],
    output_map: dict[str, Path],
    metric: str,
    quiet: bool = False,
) -> tuple[list[PairResult], list[str], list[str], float]:
    keys = sorted(set(gt_map) & set(output_map))
    missing_gt = sorted(set(output_map) - set(gt_map))
    missing_output = sorted(set(gt_map) - set(output_map))
    total_pairs = len(keys)

    results: list[PairResult] = []
    total_delta_e = 0.0
    total_pixels = 0
    start_time = time.perf_counter()

    if not quiet:
        print(f"Found {total_pairs} matched pairs. Metric={metric}", flush=True)
        if missing_gt:
            print(f"Missing gt keys: {len(missing_gt)}", flush=True)
        if missing_output:
            print(f"Missing output keys: {len(missing_output)}", flush=True)

    for index, key in enumerate(keys, start=1):
        gt_path = gt_map[key]
        output_path = output_map[key]

        gt_rgb = load_rgb_image(gt_path)
        output_rgb = load_rgb_image(output_path)

        if gt_rgb.shape != output_rgb.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': gt={gt_rgb.shape}, "
                f"output={output_rgb.shape}"
            )

        delta_e_map = compute_delta_e_map(output_rgb, gt_rgb, metric)
        pixel_count = int(delta_e_map.size)
        mean_delta_e = float(delta_e_map.mean())
        median_delta_e = float(np.median(delta_e_map))
        min_delta_e = float(delta_e_map.min())
        max_delta_e = float(delta_e_map.max())

        results.append(
            PairResult(
                key=key,
                gt_path=gt_path,
                output_path=output_path,
                mean_delta_e=mean_delta_e,
                median_delta_e=median_delta_e,
                min_delta_e=min_delta_e,
                max_delta_e=max_delta_e,
                pixel_count=pixel_count,
            )
        )

        total_delta_e += float(delta_e_map.sum())
        total_pixels += pixel_count

        if not quiet:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / index
            eta = avg_time * (total_pairs - index)
            running_mean = total_delta_e / total_pixels
            height, width = delta_e_map.shape
            print(
                f"[{index:>3d}/{total_pairs}] {key} | "
                f"size={width}x{height} | "
                f"mean={mean_delta_e:.6f} | "
                f"median={median_delta_e:.6f} | "
                f"min={min_delta_e:.6f} | "
                f"max={max_delta_e:.6f} | "
                f"running_mean={running_mean:.6f} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"eta={format_seconds(eta)}",
                flush=True,
            )

    overall_pixel_mean = total_delta_e / total_pixels if total_pixels else float("nan")
    return results, missing_gt, missing_output, overall_pixel_mean


def save_csv(results: list[PairResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "key",
                "gt_path",
                "output_path",
                "mean_delta_e",
                "median_delta_e",
                "min_delta_e",
                "max_delta_e",
                "pixel_count",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.key,
                    str(result.gt_path),
                    str(result.output_path),
                    f"{result.mean_delta_e:.6f}",
                    f"{result.median_delta_e:.6f}",
                    f"{result.min_delta_e:.6f}",
                    f"{result.max_delta_e:.6f}",
                    result.pixel_count,
                ]
            )


def print_summary(
    root: Path,
    metric: str,
    results: list[PairResult],
    missing_gt: list[str],
    missing_output: list[str],
    overall_pixel_mean: float,
    topk: int,
) -> None:
    if not results:
        print("No valid *_gt / *_output image pairs were found.", file=sys.stderr)
        return

    image_means = np.array([item.mean_delta_e for item in results], dtype=np.float64)
    sorted_results = sorted(results, key=lambda item: item.mean_delta_e, reverse=True)

    print(f"Path: {root}")
    print(f"Metric: {metric}")
    print(f"Matched pairs: {len(results)}")
    print(f"Missing gt: {len(missing_gt)}")
    print(f"Missing output: {len(missing_output)}")
    print(f"Mean Delta E (image mean average): {image_means.mean():.6f}")
    print(f"Mean Delta E (all pixels): {overall_pixel_mean:.6f}")
    print(f"Std Delta E (image mean): {image_means.std():.6f}")
    print(f"Min Delta E (image mean): {image_means.min():.6f}")
    print(f"Max Delta E (image mean): {image_means.max():.6f}")

    if topk > 0:
        print(f"\nTop {min(topk, len(sorted_results))} worst samples:")
        for result in sorted_results[:topk]:
            print(
                f"  {result.key}: mean={result.mean_delta_e:.6f}, "
                f"median={result.median_delta_e:.6f}, max={result.max_delta_e:.6f}"
            )

    if missing_gt:
        print("\nKeys missing gt:")
        for key in missing_gt:
            print(f"  {key}")

    if missing_output:
        print("\nKeys missing output:")
        for key in missing_output:
            print(f"  {key}")


def main() -> int:
    args = parse_args()
    root = args.path.expanduser().resolve()

    if not root.exists():
        print(f"Directory does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Path is not a directory: {root}", file=sys.stderr)
        return 1

    gt_map, output_map = collect_pairs(root)
    results, missing_gt, missing_output, overall_pixel_mean = evaluate_pairs(
        gt_map, output_map, args.metric, quiet=args.quiet
    )

    if not results:
        print("No valid image pairs found.", file=sys.stderr)
        return 1

    if args.save_csv is not None:
        save_csv(results, args.save_csv.expanduser().resolve())

    print_summary(
        root=root,
        metric=args.metric,
        results=results,
        missing_gt=missing_gt,
        missing_output=missing_output,
        overall_pixel_mean=overall_pixel_mean,
        topk=args.topk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
