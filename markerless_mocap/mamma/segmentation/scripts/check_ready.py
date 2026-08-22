#!/usr/bin/env python
"""Check which sequences have completed mask generation.

Scans an output directory for masks.npy files and reports which sequences
are finished vs incomplete.

Usage:
    python check_ready.py /path/to/output --expected_cams 33
    python check_ready.py /path/to/output --expected_cams 4 --show_unfinished
"""
import argparse
import glob
import os


def check_sequences(output_dir, expected_cams, show_unfinished=False, show_finished=False):
    seq_dirs = glob.glob(os.path.join(output_dir, "*", "*"))
    seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]
    seq_dirs = [d for d in seq_dirs if os.path.basename(d) != "logs"]
    seq_dirs.sort()

    if not seq_dirs:
        print(f"No sequence directories found in {output_dir}")
        return

    finished = []
    unfinished = []

    for seq_dir in seq_dirs:
        masks_paths = glob.glob(os.path.join(seq_dir, "**", "masks.npy"), recursive=True)
        dataset_name = os.path.basename(os.path.dirname(seq_dir))
        seq_name = os.path.basename(seq_dir)
        label = f"{dataset_name}/{seq_name}"

        if len(masks_paths) >= expected_cams:
            finished.append((label, len(masks_paths)))
        else:
            unfinished.append((label, len(masks_paths)))

    if show_finished and finished:
        print(f"Finished ({len(finished)}):")
        for label, count in finished:
            print(f"  {label} ({count} cameras)")

    if show_unfinished and unfinished:
        print(f"Unfinished ({len(unfinished)}):")
        for label, count in unfinished:
            print(f"  {label} ({count}/{expected_cams} cameras)")

    print(f"\nTotal: {len(seq_dirs)} sequences, "
          f"{len(finished)} finished, {len(unfinished)} unfinished")


def main():
    parser = argparse.ArgumentParser(description="Check mask generation progress")
    parser.add_argument("output_dir", help="Output directory to scan")
    parser.add_argument("--expected_cams", type=int, default=33,
                        help="Expected number of cameras (masks.npy files) per sequence (default: 33)")
    parser.add_argument("--show_unfinished", action="store_true", default=True,
                        help="List unfinished sequences (default: true)")
    parser.add_argument("--show_finished", action="store_true",
                        help="List finished sequences")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Error: {args.output_dir} is not a directory")
        return

    check_sequences(args.output_dir, args.expected_cams,
                    show_unfinished=args.show_unfinished,
                    show_finished=args.show_finished)


if __name__ == "__main__":
    main()
