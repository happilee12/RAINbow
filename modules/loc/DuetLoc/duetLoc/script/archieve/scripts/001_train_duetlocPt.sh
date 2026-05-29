DATA_ROOT=../datasets

train_alg=dagger

features=clip.b16
# ft_dim=512
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

ngpus=1
bs=4
# bs=8
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-aug.mp3d.prevalent.hm3d_gibson.envdrop.init.140k


outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}-aug.hm3d.envdrop

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
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

id=duetLocPt_v0.09_rainplusv2_onlyq_batch4
basepath=/home/master/00_WorkDir/06_DialNavPublic
project_path=/home/master/00_WorkDir/06_DialNavPublic/modules/loc/DuetLoc
output_path=/home/master/00_WorkDir/06_DialNavPublic/output
CUDA_VISIBLE_DEVICES=0 python3 main/main_nav.py $flag  \
      --tokenizer bert \
      --feat_path ${basepath}/features/clip_vit-h14_mp3d_original.hdf5 \
      --bert_ckpt_file ${basepath}/checkpoints/base/duet_vit-h14_model_step_190000.pt \
      --connectivity_dir ${basepath}/connectivity/ \
      --build_graph_maps \
      --train_data_path ${basepath}/dataset/with_dialog/train_inst.json \
      --valseen_data_path ${basepath}/dataset/with_dialog/val_seen.json \
      --valunseen_data_path ${basepath}/dataset/with_dialog/val_unseen.json \
      --model_save_path ${output_path}/DuetLoc/${id} \
      --id ${id} \
      --wandb_project debug \
      --wandb_name ${id} \
      --iterations 100000 \
      --log_every_iters 2000 \
      --wandb_log \
      --train \
      --preload_features \
      --aug_data_paths /home/master/00_WorkDir/15_ICLR_dataset/v2/aug_train_inst.jsonl \
      # --debug \
    #   --eval_first \

