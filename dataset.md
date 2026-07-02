# 📚 Dataset Description

This document describes the two dataset packages used in this codebase:

| Dataset | Directory | Granularity | Used for |
| --- | --- | --- | --- |
| **RAIN** (training) | `dataset/RAIN/` | turn / instance level | training the individual modules (navigator, question / answer generation, localization) |
| **RAIN_holistic** (evaluation) | `dataset/RAIN_holistic/` | episode level | end-to-end holistic evaluation and the DialNav Challenge scoring |

Both are derived from the same underlying RAIN episodes (a navigator reaching a
target through natural-language dialog with a guide, on Matterport3D scans).
Fields prefixed with `_` are auxiliary/internal fields.

---

## 1. RAIN — training dataset (`dataset/RAIN/`)

Turn/instance-level records: a single episode is expanded into multiple records,
one per navigation/dialog position, so modules can be trained on intermediate
states.

| File | # records | Description |
| --- | ---: | --- |
| `train.json` | 1,559 | one primary record per training episode |
| `train_inst.json` | 4,493 | training episodes expanded to per-turn instances |
| `val_seen.json` | 337 | validation (seen scans), instance level |
| `val_unseen.json` | 805 | validation (unseen scans), instance level |

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `instr_id` | str | instance id, formatted `"{episode_idx}_{chat_idx}"` (e.g. `"0_2"`) |
| `episode_idx` | int | episode index |
| `split` | str | `train` / `val_seen` / `val_unseen` |
| `scan` | str | Matterport3D scan id |
| `target` | str | target object/room name |
| `end_panos` | list[str] | goal viewpoint ids (any counts as success) |
| `start_pano` | str | start viewpoint for **this instance** |
| `nav_idx` | int | navigation step index this instance is anchored at |
| `gt_path` | list[str] | ground-truth path (viewpoints) from `start_pano` to the goal |
| `_start_pano_episode` | str | start viewpoint of the full episode |
| `_full_trajectory` | list[str] | full ground-truth navigation trajectory of the episode |
| `_full_dialog` | list[obj] | full episode dialog; each turn `{nav_idx, gui_idx, q, a}` |
| `_full_trajectory_path_length` | float | geodesic length of `_full_trajectory` *(train.json)* |
| `gt_distance` | float | shortest-path distance start→goal *(train.json)* |
| `detour_ratio` | float | `_full_trajectory_path_length / gt_distance` *(train.json)* |
| `_chat_idx` | int | dialog-turn index of this instance *(train_inst / val)* |
| `_chat_len` | int | total number of dialog turns in the episode *(train_inst / val)* |
| `nav_history` | list[str] | viewpoints already visited before this instance *(train_inst / val)* |
| `_nav_turn` | list[str] | viewpoints traversed during this navigation turn *(train_inst / val)* |

**Dialog turn** (`_full_dialog[i]`):

| Field | Type | Description |
| --- | --- | --- |
| `nav_idx` | int | navigation step at which the turn occurs |
| `gui_idx` | list[int] | guide-side reference node index/range for the turn |
| `q` | str | navigator's question (natural language) |
| `a` | str | guide's answer (natural language) |

---

## 2. RAIN_holistic — evaluation dataset (`dataset/RAIN_holistic/`)

Episode-level records (one per episode). This is the ground truth consumed by the
holistic pipeline (`holistic/script/run.sh`, `--*_anno_paths`) and by the
challenge per-sample scoring.

| File | # episodes | Description |
| --- | ---: | --- |
| `train.json` | 1,559 | train split |
| `val_seen.json` | 91 | validation, seen scans |
| `val_unseen.json` | 241 | validation, unseen scans |
| `test.json` | 285 | test split (final ranking) |

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `instr_id` | int | episode-level instruction id |
| `episode_idx` | int | episode index |
| `split` | str | `train` / `val_seen` / `val_unseen` / `test` |
| `scan` | str | Matterport3D scan id |
| `target` | str | target object/room name |
| `start_pano` | str | episode start viewpoint |
| `end_panos` | list[str] | goal viewpoint ids (any counts as success) |
| `nav_trajectory` | list[str] | ground-truth navigation trajectory (viewpoints) |
| `stop_history` | list[int] | ground-truth stop step indices |
| `dialog` | list[obj] | ground-truth dialog; each turn `{nav_idx, q, a}` |

**Dialog turn** (`dialog[i]`):

| Field | Type | Description |
| --- | --- | --- |
| `nav_idx` | int | navigation step at which the turn occurs |
| `q` | str | navigator's question (natural language) |
| `a` | str | guide's answer (natural language) |
