#!/usr/bin/env python3
"""
Batch-convert .jfif images to .jpg in a folder.

Why JPG:
- .jfif files are JPEG-family images already.
- Converting to .jpg is usually just a compatibility step.
- No extra dependency is required.

Usage:
  python convert_jfif_to_jpg.py "C:\\path\\to\\images"
  python convert_jfif_to_jpg.py "C:\\path\\to\\images" --overwrite --delete-original
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_jfif_files(folder: Path, recursive: bool) -> list[Path]:
    pattern_iter = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        [
            p
            for p in pattern_iter
            if p.is_file() and p.suffix.lower() == ".jfif"
        ]
    )


def convert_jfif_to_jpg(
    files: list[Path], overwrite: bool, delete_original: bool
) -> tuple[int, int, int]:
    converted = 0
    skipped = 0
    failed = 0

    for src in files:
        dst = src.with_suffix(".jpg")
        try:
            if dst.exists() and not overwrite:
                skipped += 1
                print(f"[SKIP] {src} -> {dst} (destination exists)")
                continue

            # JFIF is already JPEG data; copy bytes and change extension.
            shutil.copy2(src, dst)
            converted += 1
            print(f"[OK]   {src} -> {dst}")

            if delete_original:
                src.unlink()
                print(f"[DEL]  {src}")

        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {src}: {exc}")

    return converted, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert all .jfif images in a folder to .jpg."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder that contains images.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subfolders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .jpg files.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete .jfif files after successful conversion.",
    )

    args = parser.parse_args()
    folder = args.folder.resolve()
    recursive = not args.no_recursive

    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}")
        return 1

    jfif_files = find_jfif_files(folder, recursive=recursive)
    if not jfif_files:
        print(f"[INFO] No .jfif files found in: {folder}")
        return 0

    print(f"[INFO] Found {len(jfif_files)} .jfif file(s) in: {folder}")
    converted, skipped, failed = convert_jfif_to_jpg(
        jfif_files,
        overwrite=args.overwrite,
        delete_original=args.delete_original,
    )

    print("\n[SUMMARY]")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
