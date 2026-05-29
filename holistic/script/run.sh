BASEPATH=YOUR_CODE_DIRECTORY
OUTPUT_PATH=$BASEPATH/output
MODEL_CKPOINTS_PATH=$BASEPATH/dataset/checkpoints

### Model Options
## Navigation
nav_rainbow="--nav_resume_file ${MODEL_CKPOINTS_PATH}/nav_rainbow --nav_model DialogHistoryAgent --nav_act_visited_nodes"
nav_dialnav="--nav_resume_file ${MODEL_CKPOINTS_PATH}/nav_dialnav"
## Question
q_rainbow="--qg_resume_file ${MODEL_CKPOINTS_PATH}/q_rainbow"
q_dialnav="--qg_resume_file ${MODEL_CKPOINTS_PATH}/q_dialnav"
## Answer
a_rainbow="--ag_resume_file ${MODEL_CKPOINTS_PATH}/a_rainbow"
a_dialnav="--ag_resume_file ${MODEL_CKPOINTS_PATH}/a_dialnav"
## WTA
wta="--wta_mode ct_0.9"
## Localization
loc_rainbow="--loc_resume_file ${MODEL_CKPOINTS_PATH}/loc_rainbow.pth --loc_model GTL"
loc_dialnav="--loc_resume_file ${MODEL_CKPOINTS_PATH}/loc_dialnav.pt --loc_model GCN --loc_node_feats_dir ${BASEPATH}/dataset/modules/node_feats/ --loc_geodistance_nodes_path ${BASEPATH}/dataset/modules/localization/geodistance_nodes.json --loc_embedding_dir ${BASEPATH}/dataset/modules/localization/word_embeddings/"


project_id=debug
### Run with RAINbow models
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--id ${project_id} \
--output_path ${OUTPUT_PATH}/${project_id} \
--basepath ${BASEPATH} \
--connectivity_dir ${BASEPATH}/dataset/connectivity/ \
--val_seen_anno_paths ${BASEPATH}/dataset/RAIN_evaluation/val_seen.json \
--val_unseen_anno_paths ${BASEPATH}/dataset/RAIN_evaluation/val_unseen.json \
--qa_clip_tokenizer_path ${BASEPATH}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz \
--env_names val_seen,val_unseen \
$nav_rainbow \
$q_rainbow \
$wta \
$a_rainbow \
$loc_rainbow


# ### Run with DialNav models
# CUDA_VISIBLE_DEVICES=0 python3 main.py \
# --id ${project_id} \
# --output_path ${OUTPUT_PATH}/${project_id} \
# --basepath ${BASEPATH} \
# --connectivity_dir ${BASEPATH}/dataset/connectivity/ \
# --val_seen_anno_paths ${BASEPATH}/dataset/RAIN_evaluation/val_seen.json \
# --val_unseen_anno_paths ${BASEPATH}/dataset/RAIN_evaluation/val_unseen.json \
# --qa_clip_tokenizer_path ${BASEPATH}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz \
# --env_names val_seen,val_unseen \
# $nav_dialnav \
# $q_dialnav \
# $wta \
# $a_dialnav \
# $loc_dialnav
