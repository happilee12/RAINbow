<h1 align="center">RAINbow</h1>

<p align="center">
  Baselines for the <b>DialNav</b> benchmark
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.19948"><img src="https://img.shields.io/badge/arXiv-2606.19948-b31b1b" alt="arXiv"></a>
  <a href="https://happilee12.github.io/RAINbow/"><img src="https://img.shields.io/badge/Project-Page-1f72c1" alt="Project Page"></a>
  <a href="https://ead-workshop.github.io"><img src="https://img.shields.io/badge/ECCV_2026-EAD-8757e6" alt="ECCV 2026 EAD"></a>
  <a href="https://huggingface.co/spaces/lee1o21k21/DialNav-Challenge"><img src="https://img.shields.io/badge/Challenge-DialNav-1aa5b7" alt="Leaderboard"></a>
  <a href="https://github.com/happilee12/RAINbow/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-4c9a2a" alt="License MIT"></a>
</p>

> ## ⚠️ Important Note!!
>
> ### 1. The test set is released -- please submit again
>
> The test set has been newly released, and every participant must submit again. We found
> that it had not actually been released to participants.
>
> Test set download: https://drive.google.com/file/d/13GsIPcRP8sVW82PAJFSF3hDKKKzn2Nvo/view?usp=drive_link
>
> Please run your method on the newly released test set and include the resulting `test`
> split in a new submission to the
> [DialNav Challenge leaderboard](https://huggingface.co/spaces/lee1o21k21/DialNav-Challenge).
>
> **Note:** the released test set does not include the human-annotated dialog and
> navigation trajectory. Those annotations are what the score is computed from, and the
> organizers will calculate the final score offline using them.
>
> The final winner is decided on the test split only -- val_seen and val_unseen are not
> taken into account. The test split score is calculated with the
> [Evaluation Protocol](#evaluation-protocol) described below on this page.
>
> The [leaderboard](https://huggingface.co/spaces/lee1o21k21/DialNav-Challenge) shows the
> test success rate and the other reference metrics, but not the test split score. After
> the challenge period ends, we will contact the top submissions based on that score.
>
> Because the score is hidden, please check that your SR Test on the leaderboard reflects
> your own method.
>
> ### 2. For submissions made before Aug 10, 23:00 UTC
>
> Please email happilee12@korea.ac.kr with your team name. This is to ensure that your
> submission is considered for the final ranking, since the test set was not released at
> that time.
>
> ### 3. Submission deadline extended to Aug 15 AoE
>
> We know that re-running your method is extra work at short notice, and we sincerely
> apologize for the inconvenience. We have adjusted the schedule accordingly.

# 📦 Download Dataset
- RAIN training dataset: [download](https://drive.google.com/drive/folders/1Rpx1ZCrYlZvB9htLRboT88FA_-MjQbwc?usp=sharing)
- RAIN holistic dataset (for evaluation): [download](https://drive.google.com/drive/folders/1u37dzT1NbnTQwAdIo0cB7Cq1o2eT3QhK?usp=sharing)
- RAINbow dataset: [download](https://drive.google.com/drive/folders/14vyCwBVQm5glJWUu4JQVjO4-axt37kDJ?usp=sharing)
<!-- - Trained models for this codebase (`DialNav`, `RAINbow`): [download](https://drive.google.com/drive/folders/1Cbf4PkK92Wj2aTANeqfn68xvQY5nZRrF?usp=sharing)
- Pretrained weights for training: [download](https://drive.google.com/drive/folders/15JsLZqRh4VeOsFPeuieMbG7PvOlhOVKB?usp=sharing) -->

### Full download
- Download for all datasets: [download](https://drive.google.com/file/d/1J51IWSej8PdLUd9VG-H2_knxw1dv_lLY/view?usp=sharing)
- Full dataset including all above dataset, trained models, connectivity and required features.

# ⚙️ Requirements
1. Install the Matterport3D simulator.
   - Use the latest version, not v0.1.
   - After building it, set:
   ```bash
   export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
   ```

2. Create the Python environment and install dependencies.
   ```bash
   conda create --name dialnav python=3.10
   conda activate dialnav
   pip install -r requirements.txt
   ```

3. Install extra dependencies when needed.
   - For LANA:
   ```bash
   apt-get update && apt-get install -y openjdk-17-jre-headless
   ```
   - For GCN localization:
   ```python
   import nltk
   nltk.download('punkt_tab')
   ```

# 🧪 Evaluation With Provided Trained Models

Before running evaluation, make sure these folders exist under the repo root:

```text
(root)
├─ holistic/
├─ modules/
└─ dataset/
   ├─ checkpoints/
   └─ RAIN_holistic/
```

`holistic/script/run.sh` uses `YOUR_CODE_DIRECTORY`, so change it to your repo root first.
`RAINbow` results are stored under `output/rainbow`, and `DialNav` results are stored under `output/dialnav`.

```bash
cd holistic
bash script/run.sh
```

We tested this setup on 1 x RTX 3090.

### Expected Results
| Model combo | val_seen SR | val_seen DTC | val_unseen SR | val_unseen DTC | test SR | test DTC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAINbow | 58.24 | 4.66 | 29.05 | 11.08 | 20.00 | 10.08 |
| DialNav | 27.47 | 1.73 | 12.86 | 2.43 | 10.53 | 2.22 |

# 🏆 Challenge

The RAIN / DialNav benchmark is used for the **DialNav Challenge at ECCV 2026 EAD**.
Prepare a split-keyed `submission.json` (`val_seen` / `val_unseen` / `test`) and
upload it to the evaluation server.

## Evaluation Protocol

Each episode is scored with a **per-sample score** that rewards reaching the
target while using dialog efficiently:

```text
Score = E(D) × Success

E(D)  = 1 - min( max(DTC - DTC_GT, 0) / (NSC_GT - DTC_GT), 1 )
```

- **Success** — `1` if the agent stops at the target location, else `0`.
- **DTC** — Dialog Turn Count used by the agent.
- **DTC_GT** — ground-truth dialog turn count (`len(dialog)`).
- **NSC_GT** — ground-truth navigation trajectory length (`len(nav_trajectory)`).

`E(D)` stays at `1.0` while the agent asks no more than the ground-truth number of
dialog turns, and decays linearly to `0` as the excess dialog turns approach the
ground-truth navigation length. A failed episode scores `0`.

A split's score is the **mean per-sample score** over its episodes. The public
leaderboard reports **val_seen** and **val_unseen**; the **final ranking score is
determined on the `test` split** by the organizers.

## Viewing Scores

Compute the per-episode scores and per-split averages from a run's `submit.json`
and the `RAIN_holistic` ground truth (run from the repo root):

```bash
python holistic/make_per_sample_score.py \
  --submit your_sumbission_file \
  --split_dir dataset/RAIN_holistic \
  --connectivity_dir dataset/connectivity \
  --splits val_seen,val_unseen \
  --out _output/holistic/rainbow/per_sample_score.csv
```

> **Note on the `test` split.** `test.json` is **not released to participants** —
> only `val_seen` / `val_unseen` are available for local scoring. The test set is
> collected and kept privately by the organizers, and the **final challenge score is
> computed by the organizers on that held-out `test` set**.


# 🏋️ Train Each Module

Before training, prepare these folders under the repo root:

```text
(root)
├─ holistic/
├─ modules/
└─ dataset/
   ├─ pretrained/
   ├─ RAIN/
   └─ RAINbow/
```

`dataset/pretrained` contains the pretrained weights used by navigation and localization training.

## Navigator Agent

- Navigation / DST
  - RAINbow paper: [link](https://arxiv.org/pdf/2606.19948)
  - dual-strategy training with DUET
  ```bash
  cd modules/nav/DST/map_nav_src
  bash ../script/train.sh
  ```

- Navigation / ScaleVLN
  - DialNav: [link](https://happilee12.github.io/DialNav/) 
  - ScaleVLN: [link](https://scalevln.github.io/)  
  ```bash
  cd modules/nav/ScaleVLN/map_nav_src
  bash ../script/train.sh
  ```

- Question / LANA
  - LANA: [link](https://github.com/wxh1996/LANA-VLN) 
  - trained with RAIN, RAINbow question dataset
  ```bash
  cd modules/qa/LANA/finetune_src
  bash scripts/q_train.sh
  ```

## Guide Agent

- Answer / LANA
  - LANA: [link](https://github.com/wxh1996/LANA-VLN) 
  - trained with RAIN, RAINbow answer dataset
  ```bash
  cd modules/qa/LANA/finetune_src
  bash scripts/a_train.sh
  ```

- Localization / GTL
  - RAINbow paper: [link](https://arxiv.org/pdf/2606.19948)
  ```bash
  cd modules/loc/GTL/gtl
  bash script/train.sh
  ```

- Localization / GCN
  - LED : [link](https://meerahahn.github.io/way/)
  ```bash
  cd modules/loc/GCN
  bash local_script.sh
  ```

This source is based on the following repositories. Thanks for the contributions:
- [LANA](https://github.com/wxh1996/LANA-VLN)
- [ScaleVLN](https://github.com/wz0919/ScaleVLN)
- [DUET](https://github.com/cshizhe/VLN-DUET)
- [GCN](https://github.com/meera1hahn/Graph_LED)
