#!/bin/bash
# Holistic evaluation sweep: swap ONLY the navigation model checkpoint.
# All other modules (question / answer / WTA / localization) stay as the rainbow config.
set -uo pipefail

PY=/home/master/00_WorkDir/.rainbow/bin/python3
export PYTHONPATH=/home/master/00_WorkDir/Matterport3DSimulator/build:${PYTHONPATH:-}

BASEPATH=/home/master/00_WorkDir/06_RAINbow
OUTPUT_PATH=${BASEPATH}/output/holistic
MODEL_CKPOINTS_PATH=${BASEPATH}/dataset/checkpoints

# --- fixed (non-navigation) models: identical to the rainbow run ---
q_rainbow="--qg_resume_file ${MODEL_CKPOINTS_PATH}/q_rainbow"
a_rainbow="--ag_resume_file ${MODEL_CKPOINTS_PATH}/a_rainbow"
wta="--wta_mode ct_0.9"
loc_rainbow="--loc_resume_file ${MODEL_CKPOINTS_PATH}/loc_rainbow.pth --loc_model GTL"

# --- navigation checkpoints to sweep: "project_id|ckpt_path" ---
DHA=/home/master/00_WorkDir/data2/02_20250926_RAINbow/14_AdditionalModels/03_ablations_more/dha.0k
runs=(
  "dha_0k_ver3_cp1500|${DHA}/dha_0k_ver3/ckpts/cp_1500"
  "dha_0k_ver3_latest|${DHA}/dha_0k_ver3/ckpts/latest_dict"
  "dha_0k_ver4_latest|${DHA}/dha_0k_ver4/ckpts/latest_dict"
)

for entry in "${runs[@]}"; do
  project_id="${entry%%|*}"
  ckpt="${entry##*|}"
  nav="--nav_resume_file ${ckpt} --nav_model DST --nav_act_visited_nodes"

  echo "=================================================================="
  echo "[$(date '+%F %T')] RUN  ${project_id}"
  echo "  nav ckpt: ${ckpt}"
  echo "=================================================================="

  cd ${BASEPATH}/holistic
  CUDA_VISIBLE_DEVICES=0 ${PY} main.py \
    --id ${project_id} \
    --output_path ${OUTPUT_PATH}/${project_id} \
    --basepath ${BASEPATH} \
    --connectivity_dir ${BASEPATH}/dataset/connectivity/ \
    --val_seen_anno_paths ${BASEPATH}/dataset/RAIN_holistic/val_seen.json \
    --val_unseen_anno_paths ${BASEPATH}/dataset/RAIN_holistic/val_unseen.json \
    --test_anno_paths ${BASEPATH}/dataset/RAIN_holistic/test.json \
    --qa_clip_tokenizer_path ${BASEPATH}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz \
    --env_names val_seen,val_unseen,test \
    ${nav} ${q_rainbow} ${wta} ${a_rainbow} ${loc_rainbow}
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ERROR ${project_id} main.py exited $rc — skipping scoring"
    continue
  fi

  echo "[$(date '+%F %T')] SCORE ${project_id}"
  cd ${BASEPATH}
  ${PY} holistic/make_per_sample_score.py \
    --submit ${OUTPUT_PATH}/${project_id}/submit.json \
    --split_dir ${BASEPATH}/dataset/RAIN_holistic \
    --connectivity_dir ${BASEPATH}/dataset/connectivity \
    --splits val_seen,val_unseen,test \
    --out ${OUTPUT_PATH}/${project_id}/per_sample_score.csv
done

echo "[$(date '+%F %T')] ALL DONE"
