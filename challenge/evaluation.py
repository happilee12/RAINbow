# single submission 평가
# ./holistic/script/run_evaluation.sh evaluate /path/to/submission.json

# # aggregate 평가
# ./holistic/script/run_evaluation.sh aggregate /path/to/seen.json /path/to/unseen.json /path/to/test.json

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

def load_env_info(env_info_dir):
    distances_by_split_path = Path(env_info_dir) / 'shortest_distances_by_split.json'
    paths_by_split_path = Path(env_info_dir) / 'shortest_paths_by_split.json'
    metadata_by_split_path = Path(env_info_dir) / 'episode_metadata_by_split.json'

    if distances_by_split_path.exists() and paths_by_split_path.exists():
        with distances_by_split_path.open('r', encoding='utf-8') as f:
            distances_by_split = json.load(f)
        with paths_by_split_path.open('r', encoding='utf-8') as f:
            paths_by_split = json.load(f)

        shortest_distances = {}
        shortest_paths = {}
        for split_data in distances_by_split.values():
            shortest_distances.update(split_data)
        for split_data in paths_by_split.values():
            shortest_paths.update(split_data)

        if metadata_by_split_path.exists():
            with metadata_by_split_path.open('r', encoding='utf-8') as f:
                metadata_by_split = json.load(f)
            episode_metadata = {}
            for split_data in metadata_by_split.values():
                episode_metadata.update(split_data)
        else:
            episode_metadata = {}

        return shortest_distances, shortest_paths, episode_metadata

    shortest_distances_json_path = Path(env_info_dir) / 'shortest_distances.json'
    shortest_paths_json_path = Path(env_info_dir) / 'shortest_paths.json'
    episode_metadata_json_path = Path(env_info_dir) / 'episode_metadata.json'

    if not shortest_distances_json_path.exists():
        raise FileNotFoundError(f"Environment info file not found: {shortest_distances_json_path}")
    if not shortest_paths_json_path.exists():
        raise FileNotFoundError(f"Environment info file not found: {shortest_paths_json_path}")

    with shortest_distances_json_path.open('r', encoding='utf-8') as f:
        shortest_distances = json.load(f)
    with shortest_paths_json_path.open('r', encoding='utf-8') as f:
        shortest_paths = json.load(f)

    if episode_metadata_json_path.exists():
        with episode_metadata_json_path.open('r', encoding='utf-8') as f:
            episode_metadata = json.load(f)
    else:
        episode_metadata = {}

    return shortest_distances, shortest_paths, episode_metadata


def load_env_info_by_split(env_info_dir):
    distances_by_split_path = Path(env_info_dir) / 'shortest_distances_by_split.json'
    paths_by_split_path = Path(env_info_dir) / 'shortest_paths_by_split.json'
    metadata_by_split_path = Path(env_info_dir) / 'episode_metadata_by_split.json'

    if distances_by_split_path.exists() and paths_by_split_path.exists():
        with distances_by_split_path.open('r', encoding='utf-8') as f:
            shortest_distances_by_split = json.load(f)
        with paths_by_split_path.open('r', encoding='utf-8') as f:
            shortest_paths_by_split = json.load(f)

        if metadata_by_split_path.exists():
            with metadata_by_split_path.open('r', encoding='utf-8') as f:
                episode_metadata_by_split = json.load(f)
        else:
            episode_metadata_by_split = {}

        return shortest_distances_by_split, shortest_paths_by_split, episode_metadata_by_split

    shortest_distances, shortest_paths, episode_metadata = load_env_info(env_info_dir)
    return (
        {'default': shortest_distances},
        {'default': shortest_paths},
        {'default': episode_metadata},
    )

def flatten_path(path):
    if isinstance(path, list):
        flat_path = []
        for step in path:
            flat_path.extend(flatten_path(step))
        return flat_path
    return [path]


class SubmissionEvaluator:
    def __init__(self, shortest_distances, shortest_paths, episode_metadata=None, success_margin=0, error_margin=3.0):
        self.shortest_distances = shortest_distances
        self.shortest_paths = shortest_paths
        self.episode_metadata = episode_metadata or {}
        self.success_margin = success_margin
        self.error_margin = error_margin

    def _resolve_episode_info(self, item):
        instr_id = item.get('instr_id')
        metadata = None
        if instr_id is not None:
            metadata = self.episode_metadata.get(instr_id)
            if metadata is None:
                metadata = self.episode_metadata.get(str(instr_id))

        scan = item.get('scan')
        end_panos = item.get('end_panos')

        if scan is None or end_panos is None:
            if metadata is None:
                raise ValueError(
                    'Submission item is missing required episode fields and no matching '
                    f'episode metadata was found for instr_id={instr_id}. '
                    'Expected at least scan and end_panos.'
                )

        if scan is None:
            scan = metadata.get('scan')
        if end_panos is None:
            end_panos = metadata.get('end_panos')

        if scan is None or end_panos is None:
            raise ValueError(
                'Unable to resolve episode information for item. '
                f'scan={scan} end_panos={end_panos} instr_id={instr_id}'
            )

        if instr_id is not None and metadata is not None and 'scan' in item and item['scan'] != metadata.get('scan'):
            raise ValueError(
                f"Submission scan '{item['scan']}' does not match metadata scan '{metadata.get('scan')}' "
                f"for instr_id={instr_id}."
            )

        if not isinstance(end_panos, list):
            raise ValueError(f"Episode 'end_panos' must be a list for instr_id={instr_id}, got {type(end_panos)}")

        return scan, end_panos

    def _get_nearest(self, shortest_distances, goal_id, viewpoint_ids):
        near_id = viewpoint_ids[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in viewpoint_ids:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    # def _cal_dtw(self, shortest_distances, prediction, reference, success=None, threshold=None):
    #     if threshold is None:
    #         threshold = self.error_margin

    #     dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    #     dtw_matrix[0][0] = 0.0
    #     for i in range(1, len(prediction) + 1):
    #         for j in range(1, len(reference) + 1):
    #             best_previous_cost = min(
    #                 dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1]
    #             )
    #             cost = shortest_distances[prediction[i-1]][reference[j-1]]
    #             dtw_matrix[i][j] = cost + best_previous_cost

    #     dtw = dtw_matrix[len(prediction)][len(reference)]
    #     ndtw = np.exp(-dtw / (threshold * len(reference)))
    #     if success is None:
    #         success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    #     sdtw = success * ndtw
    #     return {
    #         'DTW': dtw,
    #         'nDTW': ndtw,
    #         'SDTW': sdtw,
    #     }

    # def _cal_cls(self, shortest_distances, prediction, reference, threshold=None):
    #     if threshold is None:
    #         threshold = self.error_margin
    #     def length(nodes):
    #         return np.sum([
    #             shortest_distances[a][b]
    #             for a, b in zip(nodes[:-1], nodes[1:])
    #         ])

    #     coverage = np.mean([
    #         np.exp(-np.min([
    #             shortest_distances[u][v] for v in prediction
    #         ]) / threshold)
    #         for u in reference
    #     ])
    #     expected = coverage * length(reference)
    #     score = expected / (expected + np.abs(expected - length(prediction)))
    #     return coverage * score

    def _is_success(self, scan, final_pano, end_panos):
        shortest_distances = self.shortest_distances[scan]
        for end in end_panos:
            if shortest_distances[final_pano][end] <= self.success_margin:
                return True
        return False

    def _eval_item(self, scan, predicted_path, end_panos):
        shortest_distances = self.shortest_distances[scan]
        shortest_paths = self.shortest_paths[scan]

        flattened_path = flatten_path(predicted_path)
        start_pano = flattened_path[0]
        shortest_goal = self._get_nearest(shortest_distances, start_pano, end_panos)
        shortest_path = shortest_paths[start_pano][shortest_goal]
        shortest_distance = shortest_distances[start_pano][shortest_goal]
        trajectory_distance = np.sum([
            shortest_distances[a][b]
            for a, b in zip(flattened_path[:-1], flattened_path[1:])
        ])

        scores = {
            'trajectory_lengths': trajectory_distance,
            'trajectory_steps': len(flattened_path) - 1,
            'success': float(self._is_success(scan, flattened_path[-1], end_panos)),
            'oracle_success': float(any(self._is_success(scan, x, end_panos) for x in flattened_path)),
            'spl': 0.0,
            'nav_error': np.min([shortest_distances[flattened_path[-1]][end_pano] for end_pano in end_panos]),
        }
        scores['spl'] = scores['success'] * shortest_distance / max(trajectory_distance, shortest_distance, 0.01)
        scores['gp'] = shortest_distance - scores['nav_error']
        # scores.update(self._cal_dtw(shortest_distances, flattened_path, shortest_path, scores['success']))
        # scores['CLS'] = self._cal_cls(shortest_distances, flattened_path, shortest_path)
        scores['gt_lengths'] = int(shortest_distance)

        return scores

    def eval_metrics(self, navigation_output):
        metrics = defaultdict(list)

        for item in navigation_output:
            scan, end_panos = self._resolve_episode_info(item)
            pred_path = item.get('path', [])
            details = item.get('navigation_detail', item.get('dialog', [])) or []

            if item.get('dialog') and not item.get('navigation_detail'):
                dialogs = item['dialog']
            else:
                dialogs = [d for d in details if d.get('ask', True)]

            dtc = len(dialogs)
            les = []
            # collect loc_error: always compute from 'localized_viewpoint' and 'viewpoint'
            if scan not in self.shortest_distances:
                raise FileNotFoundError(f'Environment distances for scan {scan} not found in env info')

            for detail in dialogs:
                loc_vp = detail.get('localized_viewpoint')
                gt_vp = detail.get('viewpoint')

                if loc_vp is None or gt_vp is None:
                    raise ValueError(
                        f"Missing localization info: both 'localized_viewpoint' and 'viewpoint' are required."
                        f" scan={scan} item={item.get('id', item.get('path', 'unknown'))} detail={detail}"
                    )

                try:
                    dist = float(self.shortest_distances[scan][gt_vp][loc_vp])
                except KeyError as e:
                    raise KeyError(f"Precomputed distances missing node {e} for scan {scan}")

                les.append(dist)
                detail['loc_error'] = dist

            scores = self._eval_item(scan, pred_path, end_panos)
            for k, v in scores.items():
                metrics[k].append(v)
            metrics['dtc'].append(dtc)
            metrics['le'].extend(les)

        avg_metrics = {
            'count': len(navigation_output),
            'gt_lengths': np.mean(metrics['gt_lengths']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'gp': np.mean(metrics['gp']),
            'nav_error': np.nanmean(metrics['nav_error']),
            # 'nDTW': np.mean(metrics['nDTW']) * 100,
            # 'SDTW': np.mean(metrics['SDTW']) * 100,
            # 'CLS': np.mean(metrics['CLS']) * 100,
            'le': np.mean(metrics['le']) if len(metrics['le']) > 0 else float('nan'),
            'dtc': np.mean(metrics['dtc']),
        }

        return avg_metrics, metrics


def load_predictions(prediction_file):
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)

    if isinstance(predictions, dict):
        ordered_keys = ['val_seen', 'val_unseen', 'test']
        flattened = []
        used_keys = set()

        for key in ordered_keys:
            if key in predictions:
                value = predictions[key]
                if not isinstance(value, list):
                    raise ValueError(f'Prediction split "{key}" must be a list, got {type(value)}')
                flattened.extend(value)
                used_keys.add(key)

        for key, value in predictions.items():
            if key in used_keys:
                continue
            if not isinstance(value, list):
                raise ValueError(f'Prediction split "{key}" must be a list, got {type(value)}')
            flattened.extend(value)

        return flattened

    return predictions


def load_submission(submission_file):
    with open(submission_file, 'r') as f:
        return json.load(f)


def _gt_lengths_from_metadata(metadata, instr_id=None):
    """Extract the ground-truth quantities the per-sample score needs.

    NSC_GT : length of the ground-truth navigation trajectory.
    DTC_GT : ground-truth dialog turn count.

    These must be provided by episode_metadata_by_split.json, either directly
    as integer fields (``nsc_gt`` / ``dtc_gt``) or as the ground-truth
    sequences they are derived from (``nav_trajectory`` / ``dialog``).
    """
    def as_len(value):
        return len(value) if isinstance(value, (list, tuple)) else int(value)

    nsc_gt = None
    for key in ('nsc_gt', 'NSC_GT', 'nav_trajectory', 'gt_nav_trajectory', 'gt_path'):
        if metadata.get(key) is not None:
            nsc_gt = as_len(metadata[key])
            break

    dtc_gt = None
    for key in ('dtc_gt', 'DTC_GT', 'dialog', 'gt_dialog'):
        if metadata.get(key) is not None:
            dtc_gt = as_len(metadata[key])
            break

    if nsc_gt is None or dtc_gt is None:
        raise ValueError(
            f"Episode metadata for instr_id={instr_id} is missing ground-truth "
            "fields required by the per-sample score. Provide integer "
            "'nsc_gt'/'dtc_gt', or the GT sequences 'nav_trajectory'/'dialog'."
        )

    return nsc_gt, dtc_gt


def _episode_score(dtc, dtc_gt, nsc_gt, success):
    """Per-sample score = E(D) * Success.

    E(D) = 1 - min( max(DTC - DTC_GT, 0) / (NSC_GT - DTC_GT), 1 )

    DTC is the submitted dialog turn count; DTC_GT / NSC_GT are ground truth.
    E(D) is the dialog-efficiency term: it stays 1.0 while the agent asks no
    more than the ground-truth number of turns, and decays linearly to 0 as
    the excess dialog turns approach the ground-truth navigation length.
    """
    denom = nsc_gt - dtc_gt
    if denom <= 0:
        efficiency = 0.0
    else:
        efficiency = 1.0 - min(max(dtc - dtc_gt, 0) / denom, 1.0)
    return efficiency * float(success)


def _load_avg_metrics(file_path, env_info_dir='holistic/evaluation/env_info', success_margin=0, error_margin=3.0):
    shortest_distances, shortest_paths, episode_metadata = load_env_info(env_info_dir)
    evaluator = SubmissionEvaluator(
        shortest_distances,
        shortest_paths,
        episode_metadata=episode_metadata,
        success_margin=success_margin,
        error_margin=error_margin,
    )
    predictions = load_predictions(file_path)
    avg_metrics, _ = evaluator.eval_metrics(predictions)
    return avg_metrics


def _evaluate_submission_data(
    submission,
    env_info_dir='holistic/evaluation/env_info',
    success_margin=0,
    error_margin=3.0,
):
    shortest_distances_by_split, shortest_paths_by_split, episode_metadata_by_split = load_env_info_by_split(env_info_dir)
    evaluator = SubmissionEvaluator(
        {},
        {},
        episode_metadata={},
        success_margin=success_margin,
        error_margin=error_margin,
    )

    if not isinstance(submission, dict):
        raise ValueError('Submission file must be a JSON object with split keys val_seen/val_unseen/test')

    split_aliases = {
        'val_seen': 'seen',
        'val_unseen': 'unseen',
        'test': 'test',
    }
    required_splits = list(split_aliases.keys())
    missing_splits = [split for split in required_splits if split not in submission]
    if missing_splits:
        raise ValueError(
            f'Submission file is missing required split(s): {missing_splits}. '
            f'Expected keys: {required_splits}'
        )

    def zero_scores():
        return {
            'trajectory_lengths': 0.0,
            'trajectory_steps': 0.0,
            'success': 0.0,
            'oracle_success': 0.0,
            'spl': 0.0,
            'nav_error': 0.0,
            'gp': 0.0,
            'gt_lengths': 0.0,
            'le': 0.0,
            'dtc': 0.0,
        }

    metrics_by_split = {}
    for split in required_splits:
        split_output = submission[split]
        if not isinstance(split_output, list):
            raise ValueError(f'Submission split "{split}" must be a list, got {type(split_output)}')

        env_split = split_aliases[split]
        split_metadata = episode_metadata_by_split.get(env_split, {})
        split_shortest_distances = shortest_distances_by_split.get(env_split, {})
        split_shortest_paths = shortest_paths_by_split.get(env_split, {})
        split_evaluator = SubmissionEvaluator(
            split_shortest_distances,
            split_shortest_paths,
            episode_metadata=split_metadata,
            success_margin=success_margin,
            error_margin=error_margin,
        )

        pred_by_instr = {}
        for item in split_output:
            instr_id = item.get('instr_id')
            if instr_id is None:
                raise ValueError(f'Submission item is missing instr_id in split "{split}": {item}')
            pred_by_instr[str(instr_id)] = item

        split_metrics = defaultdict(list)
        for instr_id, metadata in split_metadata.items():
            item = pred_by_instr.get(str(instr_id))
            if item is None:
                scores = zero_scores()
            else:
                scan, end_panos = split_evaluator._resolve_episode_info(item)
                pred_path = item.get('path', [])
                details = item.get('navigation_detail', item.get('dialog', [])) or []

                if item.get('dialog') and not item.get('navigation_detail'):
                    dialogs = item['dialog']
                else:
                    dialogs = [d for d in details if d.get('ask', True)]

                dtc = len(dialogs)
                les = []
                if scan not in split_evaluator.shortest_distances:
                    raise FileNotFoundError(f'Environment distances for scan {scan} not found in env info')

                for detail in dialogs:
                    loc_vp = detail.get('localized_viewpoint')
                    gt_vp = detail.get('viewpoint')

                    if loc_vp is None or gt_vp is None:
                        raise ValueError(
                            f"Missing localization info: both 'localized_viewpoint' and 'viewpoint' are required."
                            f" scan={scan} item={item.get('id', item.get('path', 'unknown'))} detail={detail}"
                        )

                    try:
                        dist = float(split_evaluator.shortest_distances[scan][gt_vp][loc_vp])
                    except KeyError as e:
                        raise KeyError(f"Precomputed distances missing node {e} for scan {scan}")

                    les.append(dist)
                    detail['loc_error'] = dist

                scores = split_evaluator._eval_item(scan, pred_path, end_panos)
                scores['le'] = float(np.mean(les)) if len(les) > 0 else 0.0
                scores['dtc'] = float(dtc)

            for key, value in scores.items():
                split_metrics[key].append(value)

            # Per-sample score = E(D) * Success, computed for the validation
            # splits only. The test-split score is intentionally NOT computed on
            # the public server; the final test-based ranking is produced
            # offline by the organizers.
            if split != 'test':
                if item is None:
                    split_metrics['score'].append(0.0)
                else:
                    nsc_gt, dtc_gt = _gt_lengths_from_metadata(metadata, instr_id)
                    split_metrics['score'].append(
                        _episode_score(dtc, dtc_gt, nsc_gt, scores['success'])
                    )

        metrics_by_split[split] = {
            'count': len(split_metadata),
            'gt_lengths': np.mean(split_metrics['gt_lengths']) if split_metrics['gt_lengths'] else 0.0,
            'steps': np.mean(split_metrics['trajectory_steps']) if split_metrics['trajectory_steps'] else 0.0,
            'lengths': np.mean(split_metrics['trajectory_lengths']) if split_metrics['trajectory_lengths'] else 0.0,
            'sr': np.mean(split_metrics['success']) * 100 if split_metrics['success'] else 0.0,
            'oracle_sr': np.mean(split_metrics['oracle_success']) * 100 if split_metrics['oracle_success'] else 0.0,
            'spl': np.mean(split_metrics['spl']) * 100 if split_metrics['spl'] else 0.0,
            'gp': np.mean(split_metrics['gp']) if split_metrics['gp'] else 0.0,
            'nav_error': np.nanmean(split_metrics['nav_error']) if split_metrics['nav_error'] else 0.0,
            'le': np.mean(split_metrics['le']) if split_metrics['le'] else 0.0,
            'dtc': np.mean(split_metrics['dtc']) if split_metrics['dtc'] else 0.0,
            'score': np.mean(split_metrics['score']) if split_metrics['score'] else 0.0,
        }

    metrics = {
        'count_seen': metrics_by_split['val_seen']['count'],
        'sr_seen': metrics_by_split['val_seen']['sr'],
        'dtc_seen': metrics_by_split['val_seen']['dtc'],
        'steps_seen': metrics_by_split['val_seen']['steps'],
        'spl_seen': metrics_by_split['val_seen']['spl'],
        'oracle_sr_seen': metrics_by_split['val_seen']['oracle_sr'],
        'score_seen': metrics_by_split['val_seen']['score'],
        'count_unseen': metrics_by_split['val_unseen']['count'],
        'sr_unseen': metrics_by_split['val_unseen']['sr'],
        'dtc_unseen': metrics_by_split['val_unseen']['dtc'],
        'steps_unseen': metrics_by_split['val_unseen']['steps'],
        'spl_unseen': metrics_by_split['val_unseen']['spl'],
        'oracle_sr_unseen': metrics_by_split['val_unseen']['oracle_sr'],
        'score_unseen': metrics_by_split['val_unseen']['score'],
        'count_test': metrics_by_split['test']['count'],
        'sr_test': metrics_by_split['test']['sr'],
        'dtc_test': metrics_by_split['test']['dtc'],
        'steps_test': metrics_by_split['test']['steps'],
        'spl_test': metrics_by_split['test']['spl'],
        'oracle_sr_test': metrics_by_split['test']['oracle_sr'],
    }

    metrics.update(_final_score(metrics))
    return metrics


def _final_score(
    metrics,
):
    """Leaderboard scoring based on the per-sample score.

    Each episode is scored as  Score = E(D) * Success  (see _episode_score).
    A split's score is the mean per-sample score over its episodes.

    The public leaderboard reports the average per-sample score on val_seen and
    val_unseen only. The test-split score is NOT computed on the public server;
    the final ranking score is the average per-sample score on the test split,
    determined offline by the organizers.
    """
    return {
        'seen_score': metrics.get('score_seen', 0.0),
        'unseen_score': metrics.get('score_unseen', 0.0),
    }


def evaluate_submission(
    submission_file,
    env_info_dir='holistic/evaluation/env_info',
    success_margin=0,
    error_margin=3.0,
):
    submission = load_submission(submission_file)
    return _evaluate_submission_data(
        submission,
        env_info_dir=env_info_dir,
        success_margin=success_margin,
        error_margin=error_margin,
    )


def dummy_evaluate(
    val_seen_file,
    val_unseen_file,
    test_file,
    env_info_dir='holistic/evaluation/env_info',
    success_margin=0,
    error_margin=3.0,
):
    submission = {
        'val_seen': load_predictions(val_seen_file),
        'val_unseen': load_predictions(val_unseen_file),
        'test': load_predictions(test_file),
    }
    return _evaluate_submission_data(
        submission,
        env_info_dir=env_info_dir,
        success_margin=success_margin,
        error_margin=error_margin,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate a submission or aggregate a split-keyed submission.json.'
    )
    parser.add_argument('--mode', choices=['evaluate', 'aggregate'], default='evaluate',
                        help='Execution mode: evaluate a single submission or aggregate split submissions.')
    parser.add_argument('--env-info-dir', type=str, default='holistic/evaluation/env_info',
                        help='Directory to read/write environment info.')
    parser.add_argument('--input-file', type=str, help='Submission JSON file to evaluate.')
    parser.add_argument('--success-margin', type=float, default=0.0, help='Success threshold in meters.')
    parser.add_argument('--error-margin', type=float, default=3.0, help='Error margin threshold for DTW and CLS metrics.')
    parser.add_argument('--dtc-score-weight', type=float, default=0.25,
                        help='Multiplier for the DTC adjustment term in the final score.')
    parser.add_argument('--output-file', type=str, help='Optional output file for average metrics or aggregated score JSON.')
    args = parser.parse_args(argv)

    if args.mode == 'evaluate':
        if args.input_file is None:
            parser.error('--mode evaluate requires --input-file')

        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_input = json.load(f)

        if isinstance(raw_input, dict) and all(split in raw_input for split in ['val_seen', 'val_unseen', 'test']):
            avg_metrics = evaluate_submission(
                args.input_file,
                env_info_dir=args.env_info_dir,
                success_margin=args.success_margin,
                error_margin=args.error_margin,
            )
            metrics = None
        else:
            shortest_distances, shortest_paths, episode_metadata = load_env_info(args.env_info_dir)
            predictions = load_predictions(args.input_file)

            evaluator = SubmissionEvaluator(
                shortest_distances,
                shortest_paths,
                episode_metadata=episode_metadata,
                success_margin=args.success_margin,
                error_margin=args.error_margin,
            )
            avg_metrics, metrics = evaluator.eval_metrics(predictions)

        print('Average metrics:')
        print(json.dumps(avg_metrics, indent=2))

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                payload = {'avg_metrics': avg_metrics}
                if metrics is not None:
                    payload['metrics'] = metrics
                json.dump(payload, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)
            print(f'Written metrics to {args.output_file}')

        return 0

    if args.mode == 'aggregate':
        if args.input_file is None:
            parser.error('--mode aggregate requires --input-file')

        result = evaluate_submission(
            args.input_file,
            env_info_dir=args.env_info_dir,
            success_margin=args.success_margin,
            error_margin=args.error_margin,
        )

        print(json.dumps(result, indent=2))
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f'Written aggregated result to {args.output_file}')

        return 0

    parser.error('Unknown mode: ' + args.mode)


if __name__ == '__main__':
    main()