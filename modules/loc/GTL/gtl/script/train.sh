basepath=YOUR_BASE_PATH
output_path=YOUR_OUTPUT_PATH
data_dir=$basepath/dataset/rain_dataset/01_rain
aug_data_dir=$basepath/dataset/rainbow/v3.1
bert_ckpt_file=$basepath/dataset/pretrained/duet_vit-h14_model_step_190000.pt   
mp3d_ft_files=$basepath/dataset/features/clip_vit-h14_mp3d_original.hdf5
connectivity=$basepath/dataset/connectivity


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


id=gtl.v3.0.00_sv

CUDA_VISIBLE_DEVICES=0 python3 main/main_nav.py $flag  \
      --tokenizer bert \
      --feat_path $mp3d_ft_files \
      --connectivity_dir $connectivity \
      --build_graph_maps \
      --train_data_path $data_dir/train_inst.json \
      --valseen_data_path $data_dir/val_seen.json \
      --valunseen_data_path $data_dir/val_unseen.json \
      --model_save_path $output_path/$id \
      --id ${id} \
      --wandb_project debug \
      --wandb_name ${id} \
      --iterations 100000 \
      --log_every_iters 2000 \
      --train \
      --preload_features \
      --aug_data_paths $aug_data_dir/aug_train_inst.jsonl \
      --bert_ckpt_file $bert_ckpt_file \
