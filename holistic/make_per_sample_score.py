"""Regenerate per_sample_score.csv from a holistic run's submit.json + dataset split.

Reproduces the per-sample RAINbow score table. All columns are recomputed from:
  - submit.json  : model output (path, dialog) per split
  - split/*.json : ground truth (nav_trajectory, dialog, meta.scan, meta.end_panos)
  - connectivity : navigation graphs, used to compute Navigation Error / success

Per-sample scoring formula:
  E(D)  = 1 - min( max(DTC - DTC_GT, 0) / (NSC_GT - DTC_GT), 1 )
  Score = E(D) * Success           (this is `score_x_success`)

Columns: split, instr_id, score, success, score_x_success,
         NSC, DTC, Navigation Error, NSC_GT, DTC_GT
where
  NSC     = len(model path)                DTC     = len(model dialog)
  NSC_GT  = len(gt nav_trajectory)         DTC_GT  = len(gt dialog)

The output file starts with a per-split average block (mean score / mean
score_x_success / count), then a blank line, then the per-sample table.

Usage:
  python holistic/make_per_sample_score.py \
      --submit   _output/holistic/rainbow/submit.json \
      --split_dir dataset/RAIN_full/split \
      --connectivity_dir dataset/connectivity \
      --out      _output/holistic/rainbow/per_sample_score.csv
  # restrict to specific splits:
  python holistic/make_per_sample_score.py --splits val_seen,val_unseen
"""
import os
import sys
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import Evaluator

FIELDS = ['split', 'instr_id', 'score', 'success', 'score_x_success',
          'NSC', 'DTC', 'Navigation Error', 'NSC_GT', 'DTC_GT']


def flatten(path):
    """Flatten a possibly-nested trajectory into a flat list of viewpoint ids."""
    flat = []
    for step in path:
        if isinstance(step, list):
            flat.extend(flatten(step))
        else:
            flat.append(step)
    return flat


def episode_score(dtc, dtc_gt, nsc_gt):
    """E(D) = 1 - min( max(DTC - DTC_GT, 0) / (NSC_GT - DTC_GT), 1 )."""
    denom = nsc_gt - dtc_gt
    if denom <= 0:
        # Degenerate ground truth (no navigable steps beyond dialog turns).
        return 0
    # NOTE: `1 - min(ratio, 1)` with an int literal `1` keeps a fully-clamped
    # score as int 0 (round(0, n) -> 0), matching the original CSV formatting;
    # a zero ratio yields float 1.0 and partial penalties yield floats.
    return 1 - min(max(dtc - dtc_gt, 0) / denom, 1)


def load_ground_truth(split_dir, splits):
    """instr_id -> gt record, per split."""
    gt = {}
    for sp in splits:
        path = os.path.join(split_dir, f'{sp}.json')
        gt[sp] = {int(x['instr_id']): x for x in json.load(open(path))}
    return gt


def build_rows(submit, gt, evaluator, splits, ndigits=6, success_margin=0.0):
    """Compute the per-sample score rows for the requested splits."""
    rows = []
    for sp in splits:
        for it in submit[sp]:
            iid = int(it['instr_id'])
            g = gt[sp][iid]
            meta = g['meta']

            flat_path = flatten(it['path'])
            nsc = len(flat_path)
            dtc = len(it['dialog'])
            nsc_gt = len(g['nav_trajectory'])
            dtc_gt = len(g['dialog'])

            nav_error = float(evaluator.get_shortest(meta['scan'], flat_path[-1], meta['end_panos']))
            success = 1 if nav_error <= success_margin else 0

            score = round(episode_score(dtc, dtc_gt, nsc_gt), ndigits)
            rows.append({
                'split': sp,
                'instr_id': iid,
                'score': score,
                'success': success,
                'score_x_success': float(score) * success,
                'NSC': nsc,
                'DTC': dtc,
                'Navigation Error': round(nav_error, ndigits),
                'NSC_GT': nsc_gt,
                'DTC_GT': dtc_gt,
            })
    return rows


def split_averages(rows, splits, ndigits=6):
    """Per-split mean score / mean score_x_success / count."""
    summary = []
    for sp in splits:
        sub = [r for r in rows if r['split'] == sp]
        if not sub:
            continue
        n = len(sub)
        mean_score = sum(float(r['score']) for r in sub) / n
        mean_sxs = sum(float(r['score_x_success']) for r in sub) / n
        summary.append({
            'split': sp,
            'count': n,
            'mean_score': round(mean_score, ndigits),
            'mean_score_x_success': round(mean_sxs, ndigits),
        })
    return summary


def write_output(out, rows, splits, ndigits=6):
    """Write the average block, a blank line, then the per-sample table."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    summary = split_averages(rows, splits, ndigits)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        # --- per-split average block ---
        w.writerow(['split', 'count', 'mean_score', 'mean_score_x_success'])
        for s in summary:
            w.writerow([s['split'], s['count'], s['mean_score'], s['mean_score_x_success']])
        # --- blank separator line ---
        f.write('\n')
        # --- per-sample table ---
        dw = csv.DictWriter(f, fieldnames=FIELDS)
        dw.writeheader()
        dw.writerows(rows)
    return summary


def generate(submit_path, split_dir, connectivity_dir, out, splits=None,
             success_margin=0.0, error_margin=3.0, ndigits=6):
    """End-to-end: load inputs, build rows, write output. Returns (rows, summary)."""
    submit = json.load(open(submit_path))
    if splits is None:
        splits = list(submit.keys())
    else:
        missing = [sp for sp in splits if sp not in submit]
        if missing:
            raise KeyError(f'splits not present in {submit_path}: {missing}')

    gt = load_ground_truth(split_dir, splits)
    scans = sorted({gt[sp][int(it['instr_id'])]['meta']['scan']
                    for sp in splits for it in submit[sp]})
    evaluator = Evaluator(connectivity_dir, scans,
                          success_margin=success_margin, error_margin=error_margin)

    rows = build_rows(submit, gt, evaluator, splits, ndigits, success_margin)
    summary = write_output(out, rows, splits, ndigits)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit', default='_output/holistic/rainbow/submit.json')
    ap.add_argument('--split_dir', default='dataset/RAIN_full/split')
    ap.add_argument('--connectivity_dir', default='dataset/connectivity')
    ap.add_argument('--out', default='_output/holistic/rainbow/per_sample_score.csv')
    ap.add_argument('--splits', default='val_seen,val_unseen',
                    help='Comma-separated splits to include')
    ap.add_argument('--success_margin', type=float, default=0.0,
                    help='A sample is a success when Navigation Error <= this (meters).')
    ap.add_argument('--error_margin', type=float, default=3.0)
    ap.add_argument('--round', type=int, default=6, dest='ndigits',
                    help='Decimal places for float columns.')
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',')] if args.splits else None
    rows, summary = generate(args.submit, args.split_dir, args.connectivity_dir,
                             args.out, splits=splits,
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
