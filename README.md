# 🌐 Project page: [link](https://happilee12.github.io/RAINbow/)

# 📦 Download Dataset
- RAIN training dataset: [download](https://drive.google.com/drive/folders/1Rpx1ZCrYlZvB9htLRboT88FA_-MjQbwc?usp=sharing)
- RAIN evaluation dataset: [download](https://drive.google.com/drive/folders/1u6LHI90UbXSevdw8uYIHTdc6qm7in_bc?usp=sharing)
- RAINbow dataset: [download](https://drive.google.com/drive/folders/14vyCwBVQm5glJWUu4JQVjO4-axt37kDJ?usp=sharing)
<!-- - Trained models for this codebase (`DialNav`, `RAINbow`): [download](https://drive.google.com/drive/folders/1Cbf4PkK92Wj2aTANeqfn68xvQY5nZRrF?usp=sharing)
- Pretrained weights for training: [download](https://drive.google.com/drive/folders/15JsLZqRh4VeOsFPeuieMbG7PvOlhOVKB?usp=sharing) -->

### Full download
- Download for all datasets: [download](https://drive.google.com/file/d/11i4eqslpxZerIhrZqB-S-CYQYnwljSEQ/view?usp=sharing)
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
   └─ RAIN_evaluation/
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
