
train_alg=dagger

features=clip.b16
# ft_dim=512
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=4
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-aug.mp3d.prevalent.hm3d_gibson.envdrop.init.140k


# outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}-aug.hm3d.envdrop

flag="
      --dataset r2r
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size ${bs}
      --lr 1e-5
      --iters 200000
      --log_every 500
      --aug_times 9

      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."


gpus=3
id=duetloc.v1.4.02_ft
basepath=/home/cvlab10/project/VLN/DialNav2
project_path=/home/cvlab10/project/VLN/DialNav2/modules/loc/DuetLoc
output_path=/home/cvlab10/project/VLN/dataset/DialNav2_outputs/DuetLoc
CUDA_VISIBLE_DEVICES=$gpus python3 main/main_nav.py $flag  \
      --tokenizer bert \
      --feat_path ${basepath}/dataset/features/clip_vit-h14_mp3d_original.hdf5 \
      --bert_ckpt_file ${basepath}/dataset/checkpoints/base/duet_vit-h14_model_step_190000.pt \
      --connectivity_dir ${basepath}/dataset/connectivity/ \
      --build_graph_maps \
      --train_data_path ${basepath}/dataset/rain_dataset/01_rain/train_inst.json \
      --valseen_data_path ${basepath}/dataset/rain_dataset/01_rain/val_seen.json \
      --valunseen_data_path ${basepath}/dataset/rain_dataset/01_rain/val_unseen.json \
      --model_save_path ${output_path}/DuetLoc/${id} \
      --id ${id} \
      --wandb_project ICLR2026 \
      --wandb_name ${id} \
      --iterations 50000 \
      --log_every_iters 2000 \
      --wandb_log \
      --eval_first \
      --train \
      --resume_file /home/cvlab10/project/VLN/dataset/iclr_models/v1.4_ft/duetloc.v1.4.01_pt/latest.pth \
      # --debug \
      # --aug_data_paths /home/cvlab10/project/VLN/DialNav2/dataset/rainplus/rainplus_v1.4/aug_train_inst.jsonl \
