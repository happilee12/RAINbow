basepath=YOUR_BASE_PATH
output_path=YOUR_OUTPUT_PATH
data_dir=$basepath/dataset/rain_dataset/01_rain
aug_data_dir=$basepath/dataset/rainbow/v3.1
bert_ckpt_file=$basepath/dataset/pretrained/duet_vit-h14_model_step_190000.pt   
mp3d_ft_files=$basepath/dataset/features/clip_vit-h14_mp3d_original.hdf5
val_ft_file=$basepath/dataset/features/clip_vit-h14_mp3d_original.hdf5
connectivity=$basepath/dataset/connectivity

features=clip.h14
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768
ngpus=1

seed=0
max_action_len=50
max_instr_len=200

batch_size=8
its=200000
log_every=1000

DATA_ROOT=YOUR_DATA_ROOT_PATH
flag="--root_dir ${DATA_ROOT}
      --dataset cvdn
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      
      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --expert_policy spl
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      --max_action_len ${max_action_len} 
      --max_instr_len ${max_instr_len} 
      --batch_size ${batch_size}
      --lr 1e-5
      --iters ${its}
      --log_every ${log_every}
      --env_aug
      --optim adamW
      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --ml_weight 0.15
      --feat_dropout 0.4
      --dropout 0.5
      --gamma 0."


id=debug
CUDA_VISIBLE_DEVICES=0 python3 -m dst.main_nav $flag  \
      --id $id \
      --tokenizer bert \
      --mp3d_ft_files $mp3d_ft_files \
      --val_ft_file $val_ft_file \
      --connectivity_dir $connectivity \
      --bert_ckpt_file $bert_ckpt_file \
      --output_dir $output_path/$id \
      --data_dir $data_dir \
      --max_action_len 50 \
      --eval_first \
      --validation_set_history \
      --disactivate_stop_node_jump \
      --act_visited_nodes \
      --iters 100000 \
      --batch_size 4 \
      --grad_accum_steps 16 \
      --log_every 100 \
      --lr 1e-4 \
      --aug_data_dir $aug_data_dir \
      --aug \
      --aug_instance_training_times 9 \
      --instance_training_times 1 \
      --seed 3 \
