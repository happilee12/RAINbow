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

id=${1:-"id"}
resume_file=${2:-"none"}
wandb_log=${3:-"none"}
wandb_project=${4:-"wandb_project"}
basepath=${5:-"basepath"}
output_path=${6:-"output_path"}

cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 main/main_nav.py $flag"
cmd="$cmd --tokenizer bert"
cmd="$cmd --feat_path ${basepath}/dataset/features/clip_vit-h14_mp3d_original.hdf5"
cmd="$cmd --bert_ckpt_file ${basepath}/dataset/checkpoints/base/duet_vit-h14_model_step_190000.pt"
cmd="$cmd --connectivity_dir ${basepath}/dataset/connectivity/"
cmd="$cmd --build_graph_maps"
cmd="$cmd --train_data_path ${basepath}/dataset/rain_dataset/with_dialog/train_inst.json"
cmd="$cmd --valseen_data_path ${basepath}/dataset/rain_dataset/with_dialog/val_seen.json"
cmd="$cmd --valunseen_data_path ${basepath}/dataset/rain_dataset/with_dialog/val_unseen.json"
cmd="$cmd --model_save_path ${output_path}/${id}"
cmd="$cmd --id ${id}"
cmd="$cmd --wandb_project $wandb_project"
cmd="$cmd --wandb_name ${id}"
cmd="$cmd --preload_features"
cmd="$cmd --log_every_iters 2000"
cmd="$cmd --wandb_log"
cmd="$cmd --resume_file $resume_file"
# cmd="$cmd --iterations 100000"
cmd="$cmd --eval_first"

# Add conditional arguments
if [ "$wandb_log" = "--wandb_log" ]; then
    cmd="$cmd --wandb_log"
fi

if [ -n "$resume_file" ] && [ "$resume_file" != "none" ]; then
    cmd="$cmd --resume_file $resume_file"
fi

# Execute the command
echo "Executing: $cmd"
eval $cmd