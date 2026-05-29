ob_type=pano
feedback=sample
ft_dim=512
ngpus=1

basepath=/home/master/00_WorkDir/05_DialNavPublic/DialNavHolistic
output_path=/home/master/00_WorkDir/06_RAINbow/output/a_lana
connectivity_dir=${basepath}/dataset/connectivity/
img_ft_file=${basepath}/dataset/features/CLIP-ViT-B-16-views.tsv
bpe_path=${basepath}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz
anno_dir=${basepath}/dataset/rain_dataset/01_rain
aug_data_dir=${basepath}/dataset/rainbow/v3.1
resume_file=/home/master/00_WorkDir/06_RAINbow/dataset/pretrained/lana-caption-pretrain-max200

id=debug

flag="
  --rl_teacher_weight 0.4
  --root_dir ../datasets
  --ob_type ${ob_type}
  --world_size ${ngpus}
  --num_l_layers 0
  --num_x_layers 4
  --hist_enc_pano
  --hist_pano_num_layers 2
  --feedback ${feedback}
  --image_feat_size ${ft_dim}
  --angle_feat_size 4
  --lr 1e-4
  --iters 100000
  --log_every 2000
  --batch_size 32
  --target_batch_size 128
  --optim adamW
  --ml_weight 0.2
  --feat_dropout 0.4
  --dropout 0.5
  --clip_lr 1e-5
  --vln_task_weight 5
  --caption_task_weight 1"

CUDA_VISIBLE_DEVICES=0 python3 r2r/main.py $flag \
  --act_pred_token ob \
  --bpe_path $bpe_path \
  --output_dir $output_path/$id \
  --connectivity_dir $connectivity_dir \
  --img_ft_file $img_ft_file \
  --anno_dir $anno_dir \
  --task_name caption \
  --id $id \
  --dataset RAIN-qa \
  --max_given_len 0 \
  --features clip16 \
  --use_clip16 \
  --eval_first \
  --caption_type answer \
  --caption_target_path gt_path \
  --max_action_len 20 \
  --max_instr_len 200 \
  --use_cache \
  --cache_type gpu \
  --aug_times 9 \
  --aug_data_dir $aug_data_dir \
  --aug \
  --resume_file $resume_file \
  # --aug_use_cace \
