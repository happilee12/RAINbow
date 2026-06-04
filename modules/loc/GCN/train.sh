basepath=YOUR_BASE_PATH
output_path=YOUR_OUTPUT_PATH
embedding_dir=YOUR_EMBEDDING_DIR
connectivity_dir=${basepath}/dataset/connectivity/
img_ft_file=${basepath}/dataset/features/CLIP-ViT-B-16-views.tsv
bpe_path=${basepath}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz
anno_dir=${basepath}/dataset/rain_dataset/01_rain
aug_data_dir=${basepath}/dataset/rainbow/v3.1
panofeat_dir=${basepath}/dataset/modules/node_feats/
geodistance_file=${basepath}/dataset/modules/localization/geodistance_nodes.json
checkpoint_dir=$output_path/checkpoints/
predictions_dir=$output_path/predictions/
train_files=$basepath/dataset/RAIN/instances/train_inst.json
val_seen_file=$basepath/dataset/RAIN/instances/val_seen.json
val_unseen_file=$basepath/dataset/RAIN/instances/val_unseen.json
test_file=$basepath/dataset/RAIN/instances/test.json


id=gcn.v0.01
CUDA_VISIBLE_DEVICES='0' python3 -m src.cross_modal.run \
--name $id \
--model_save \
--gcn \
--data_dir None \
--connect_dir $connectivity_dir \
--panofeat_dir $panofeat_dir \
--embedding_dir $embedding_dir \
--geodistance_file $geodistance_file \
--checkpoint_dir $checkpoint_dir \
--predictions_dir $predictions_dir \
--max_nodes 345 \
--max_nodes_test 345 \
--train_files $train_files \
--val_seen_file $val_seen_file \
--val_unseen_file $val_unseen_file \
--test_file $test_file \
--num_gcn_layers 3 \
--num_epoch 1 \
--early_stopping 10 \
--train \
--num_epoch 50 \
# --wandb_log \
# --eval_ckpt /home/master/00_WorkDir/data/02_pt/gcn.v4.0.00_pt/best_unseen_LE.pt
