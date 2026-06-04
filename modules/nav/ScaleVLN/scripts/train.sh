basepath=YOUR_BASE_PATH
output_path=YOUR_OUTPUT_PATH
data_dir=$basepath/dataset/rain_dataset/01_rain
aug_data_dir=$basepath/dataset/rainbow/v3.1
bert_ckpt_file=$basepath/dataset/pretrained/duet_vit-h14_model_step_190000.pt   
mp3d_ft_files=$basepath/dataset/features/clip_vit-h14_mp3d_original.hdf5
val_ft_file=$basepath/dataset/features/clip_vit-h14_mp3d_original.hdf5
connectivity=$basepath/dataset/connectivity


train_alg=dagger
features=clip.h14
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

org_its=100000
org_log_every=500

ngpus=1
batch_size=4
# seed=0
max_action_len=50
max_instr_len=200

org_bs=8
grad_accum_steps=$(expr $org_bs / $batch_size) # 16 / bs

flag="--dataset cvdn
      --world_size ${ngpus}
      --tokenizer bert      
      --grad_accum_steps ${grad_accum_steps}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len ${max_action_len} 
      --max_instr_len ${max_instr_len} 

      --batch_size ${batch_size}
      --lr 1e-5
      --aug_times 9

      --env_aug

      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."


id=SV
CUDA_VISIBLE_DEVICES='0' python3 nav_agent/main_nav.py $flag  \
      --id $id \
      --output_dir $output_path/$id \
      --tokenizer bert \
      --data_dir $data_dir \
      --bert_ckpt_file $bert_ckpt_file \
      --mp3d_ft_files $mp3d_ft_files \
      --val_ft_file $val_ft_file \
      --connectivity_dir $connectivity \
      --except_train_seen \
      --eval_first \
      --grad_accum_steps 2 \ 
      --iters 200000 \
      --log_every 1000 \

wta_id=SV_wta
CUDA_VISIBLE_DEVICES='0' python3 nav_agent/main_nav.py $flag  \
      --id $id \
      --output_dir $output_path/$wta_id \
      --tokenizer bert \
      --resume_file $output_path/$id/ckpts/latest_dict \
      --data_dir $data_dir \
      --bert_ckpt_file $bert_ckpt_file \
      --mp3d_ft_files $mp3d_ft_files \
      --val_ft_file $val_ft_file \
      --connectivity_dir $connectivity \
      --except_train_seen \
      --extend_wta \
      --grad_accum_steps 2 \
      --iters 10000 \
      --log_every 1000 \
