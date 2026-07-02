"""Regenerate the per-sample RAINbow score table for the VALIDATION splits only
(val_seen, val_unseen).

Identical output format to make_per_sample_score.py -- a per-split average block,
a blank line, then the per-sample table -- but restricted to the two validation
splits (test is excluded). Shares all scoring logic with make_per_sample_score.

Usage:
  python holistic/make_per_sample_score_val.py \
      --submit   _output/holistic/rainbow/submit.json \
      --split_dir dataset/RAIN_full/split \
      --connectivity_dir dataset/connectivity \
      --out      _output/holistic/rainbow/per_sample_score_val.csv
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_per_sample_score import generate

VAL_SPLITS = ['val_seen', 'val_unseen']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit', default='_output/holistic/rainbow/submit.json')
    ap.add_argument('--split_dir', default='dataset/RAIN_full/split')
    ap.add_argument('--connectivity_dir', default='dataset/connectivity')
    ap.add_argument('--out', default='_output/holistic/rainbow/per_sample_score_val.csv')
    ap.add_argument('--success_margin', type=float, default=0.0,
                    help='A sample is a success when Navigation Error <= this (meters).')
    ap.add_argument('--error_margin', type=float, default=3.0)
    ap.add_argument('--round', type=int, default=6, dest='ndigits',
                    help='Decimal places for float columns.')
    args = ap.parse_args()

    rows, summary = generate(args.submit, args.split_dir, args.connectivity_dir,
                             args.out, splits=VAL_SPLITS,
                             success_margin=args.success_margin,
                             error_margin=args.error_margin, ndigits=args.ndigits)

    print(f'Wrote {len(rows)} rows to {args.out}')
    print('Per-split averages:')
    for s in summary:
        print(f"  {s['split']:<11} count={s['count']:<4} "
              f"mean_score={s['mean_score']:.6f}  "
              f"mean_score_x_success={s['mean_score_x_success']:.6f}")


if __name__ == '__main__':
    main()
