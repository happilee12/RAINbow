import os
import json
import time
import numpy as np
from collections import defaultdict
import logging

import torch
# import wandb

import sys
sys.path.append("..")
sys.path.append(".")

from utils.misc import set_random_seed
# from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, ImageFeaturesDB2
# from main.data_utils import construct_instrs
from main.env import EnvBatch, GraphEnvBatch, R2RNavBatch
from main.parser import parse_args

from models.vlnbert_init import get_tokenizer
from main.graph_agent import GraphVlnAgent

from main.data_loader import VLNDialogDataset
from tqdm import tqdm
seen_scans = ["7y3sRwLe3Va", "PX4nDJXEHrG", "1pXnuDYAj8r", "D7G3Y4RVNrH", "SN83YJsR3w2", "i5noydFURQK", "8WUmhLawc2A", "29hnd4uzFmX", "S9hNv5qa7GM", "EDJbREhghzL", "p5wJjkQkbXX", "E9uDoFAP3SH", "VzqfbhrpDEA", "PuKPg4mmafe", "jh4fc5c5qoQ", "B6ByNegPMKs", "Vvot9Ly1tCj", "5q7pvUzZiYa", "GdvgFV5R1Z5", "Pm6F8kyY3z2", "Uxmj2M2itWa", "qoiz87JEwZ2", "759xd9YjKW5", "r1Q1Z4BcV1o", "JmbYfDe2QKZ", "gZ6f7yhEvPG", "uNb9QFRL6hY", "VFuaQ6m2Qom", "2n8kARJN3HM", "dhjEzFoUFzH", "V2XKFyX4ASd", "VLzqgDo317F", "XcA2TqTSSAj", "1LXtFkjw3qL", "17DRP5sb8fy", "cV4RVeZvu5T", "JF19kD82Mey", "HxpKQynjfin", "pRbA3pwrgk9", "mJXqzFtmKg4", "ZMojNkEp431", "aayBHfsNo7d", "b8cTxDM8gDG", "ac26ZMwG7aT", "JeFG25nYj2p", "vyrNrziPKCB", "D7N2EKCX4Sj", "sT4fr6TAbpF", "5LpN3gDmAk7", "s8pcmisQ38h", "e9zR4mvMWw7", "r47D5H71a5s", "ur6pFq6Qu1A", "YmJkqBEsHnH", "rPc6DW4iMge", "kEZ7cmS4wCh", "VVfe2KiqLaN", "sKLMLpTHeUy", "ULsKaCPVFJR", "gTV8FGcVJC9", "82sE5b5pLXE"]
def _preload_features(data_loader, agent, split="seen"):
    """
    Preload features for all unique scan+viewpoint pairs in the data_loader if needed.
    """
    all_scan_viewpoints = set()
    agent.env._load_nav_graphs( seen_scans if split == "seen" else data_loader.scans)
    for i in range(len(data_loader)):
        item = data_loader[i]
        all_scan_viewpoints.add((item['scanName'], item['viewpointId']))
        conn_graph = agent.env.graphs[item['scanName']]
        if item['viewpointId'] in conn_graph:
            for neighbor in conn_graph[item['viewpointId']]:
                all_scan_viewpoints.add((item['scanName'], neighbor))
    print(f"Preloading features for {len(all_scan_viewpoints)} viewpoints...")
    agent.env.feat_db.preload_features(list(all_scan_viewpoints))

def setup_logging(args):
    """
    Setup logging to both file and console.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(args.model_save_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(log_dir, f'{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    
    return logger, log_file

def log_metrics(logger, metrics, prefix="", iteration=None):
    """
    Log metrics to both wandb and file.
    """
    # Log to file
    if iteration is not None:
        logger.info(f"{prefix} Iteration {iteration}: {metrics}")
    else:
        logger.info(f"{prefix}: {metrics}")
    
    # Log to wandb if enabled
    if hasattr(logger, 'wandb_enabled') and logger.wandb_enabled:
        wandb.log(metrics)

def build_env(connectivity_dir, ft_file, image_feat_size, batch_size, angle_feat_size, use_gpu=True, preload_all=False):
    feat_db = ImageFeaturesDB(ft_file, image_feat_size, use_gpu=use_gpu, preload_all=preload_all)
    env = GraphEnvBatch(connectivity_dir, feat_db=feat_db, batch_size=batch_size, angle_feat_size=angle_feat_size)
    return env


def load_data(data_dir,  load_neighbor=False, connectivity_dir=None, debug=False):
    tokenizer = get_tokenizer()
    data_loader = VLNDialogDataset(data_dir, tokenizer=tokenizer, max_dialog_len=512, load_neighbor=load_neighbor, connectivity_dir=connectivity_dir, debug=debug)
    return data_loader


def evaluate(agent, scans, predicted, gt):
    distances = [agent.env.shortest_distances[scan_id][predicted[idx]][gt[idx]] for idx, scan_id in enumerate(scans)]
    correct_0m = [1 for i in range(len(distances)) if distances[i] == 0]
    correct_3m = [1 for i in range(len(distances)) if distances[i] <= 3]
    correct_5m = [1 for i in range(len(distances)) if distances[i] <= 5]
    correct_10m = [1 for i in range(len(distances)) if distances[i] <= 10]
    return {
        "LE": distances,
        "0mAcc": correct_0m,
        "3mAcc": correct_3m,
        "5mAcc": correct_5m,
        "10mAcc": correct_10m
    }
  

def valid_loc(args, data_loader, agent, batch_size=4, preload_features=True, split_type="seen"):
    """
    Build complete graph maps in batch for the locations in the data loader.
    """
    print(f"Building complete graph maps for {len(data_loader)} locations...")
    if preload_features:
        print("split_type", split_type)
        _preload_features(data_loader, agent, split_type)
    predicted_list = []
    gt_list = []

    results = {
        "LE": [],
        "0mAcc": [],
        "3mAcc": [],
        "5mAcc": [],
        "10mAcc": []
    }
    
    for batch_idx in tqdm(range(0, len(data_loader), batch_size), desc="Processing batches", total=(len(data_loader)-1)//batch_size + 1):
        # print(f"Processing batch {batch_idx} to {batch_idx+batch_size}...")
        batch_items = data_loader[batch_idx:batch_idx+batch_size]
        predicted, gt = agent.test(batch_items, cache_graph_enc=False, save_gmap_path=None)
        scans = [item['scanName'] for item in batch_items]
        gt_list.extend(gt)
        eval_result = evaluate(agent, scans, predicted, gt)
        results["LE"].extend(eval_result["LE"])
        results["0mAcc"].extend(eval_result["0mAcc"]) 
        results["3mAcc"].extend(eval_result["3mAcc"])
        results["5mAcc"].extend(eval_result["5mAcc"])
        results["10mAcc"].extend(eval_result["10mAcc"])

        torch.cuda.empty_cache()
        predicted_list.extend(predicted)
        data_size = len(predicted_list)

        # ret = {
        #     "LE": sum(results["LE"])/data_size,
        #     "0mAcc": sum(results["0mAcc"])/data_size,
        #     "3mAcc": sum(results["3mAcc"])/data_size,
        #     "5mAcc": sum(results["5mAcc"])/data_size,
        #     "10mAcc": sum(results["10mAcc"])/data_size,
        # }

        # if batch_idx % 10 == 0:
        #     tqdm.write(f"finished {len(predicted_list)} items: {ret}")
    ret = {
        "LE": sum(results["LE"])/data_size,
        "0mAcc": sum(results["0mAcc"])/data_size,
        "3mAcc": sum(results["3mAcc"])/data_size,
        "5mAcc": sum(results["5mAcc"])/data_size,
        "10mAcc": sum(results["10mAcc"])/data_size,
    }

    return ret


def train_one_epoch(args, data_loader, agent, batch_size=4, preload_features=True, epoch=0):
    """
    Train the agent to predict the correct viewpointId for each item in the data_loader.
    """
    print(f"Training agent for {len(data_loader)} locations...")
    if preload_features:
        _preload_features(data_loader, agent)
    total_loss = 0.0
    num_batches = 0
    predicted_list = []
    gt_list = []
    
    # Shuffle the data
    indices = list(range(len(data_loader)))
    np.random.shuffle(indices)
    
    for batch_idx in tqdm(range(0, len(data_loader), batch_size), desc="Training batches", total=(len(data_loader)-1)//batch_size + 1):
        batch_indices = indices[batch_idx:batch_idx+batch_size]
        batch_items = [data_loader[i] for i in batch_indices]
        loss, predicted, gt = agent.train_step(batch_items)
        total_loss += loss
        num_batches += 1
        predicted_list.extend(predicted)
        gt_list.extend(gt)
        accuracy = sum([1 for i in range(len(predicted_list)) if predicted_list[i] == gt_list[i]]) / len(predicted_list)
        tqdm.write(f"Batch {batch_idx//batch_size + 1}/{(len(data_loader)-1)//batch_size + 1} - Loss: {loss:.4f}")
        tqdm.write(f"Accuracy ({len(predicted_list)} items): {accuracy:.4f}")
        torch.cuda.empty_cache()
        # wandb log batch
        if args.wandb_log:
            wandb.log({
                'train/batch_loss': loss,
                'train/batch_accuracy': accuracy,
                'train/batch_idx': batch_idx//batch_size + 1,
                'train/epoch': epoch+1
            })
    epoch_accuracy = sum([1 for i in range(len(predicted_list)) if predicted_list[i] == gt_list[i]]) / len(predicted_list)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # wandb log epoch
    if args.wandb_log:
        wandb.log({
            'train/epoch_loss': avg_loss,
            'train/epoch_accuracy': epoch_accuracy,
            'train/epoch': epoch+1
        })
    return avg_loss, epoch_accuracy

def train_iteration_based(args, train_data_loader, aug_train_data_loader, val_loader, agent, batch_size=4, preload_features=True, max_iterations=1000, validation_interval=100, start_iter=0):
    """
    Train the agent iteration by iteration instead of epoch by epoch.
    This is more suitable for large datasets.
    Uses aug_train and train data in 9:1 ratio.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training agent for {max_iterations} iterations with validation every {validation_interval} iterations...")
    logger.info(f"Starting from iteration {start_iter}, target iteration: {start_iter+max_iterations}")
    logger.info(f"Using aug_train:train ratio of 9:1")

    total_loss = 0.0
    iteration = start_iter
    predicted_list = []
    acc_iterations = 0
    gt_list = []
    
    # Create data iterators for both datasets
    train_indices = list(range(len(train_data_loader)))
    np.random.shuffle(train_indices)

    if aug_train_data_loader:
        aug_train_indices = list(range(len(aug_train_data_loader)))
        np.random.shuffle(aug_train_indices)
    
    aug_train_idx = 0
    train_idx = 0
    
    best_results = {
        "seen": {
            # "3mAcc": -1,
            "LE": 1000,  # Add LE for seen split
        },
        "unseen": {
            # "3mAcc": -1,
            "LE": 1000,
        }
    }
    
    # Track best average LE across seen and unseen
    best_avg_LE = 1000.0
    
    if preload_features:
        _preload_features(train_data_loader, agent, split="seen")
    
    one_iteration = 1 + (args.aug_times if aug_train_data_loader else 0)
    for iteration in tqdm(range(start_iter, start_iter+max_iterations, one_iteration), desc="Training iterations"):
        
        # Train with train data (1 iteration)
        for _ in range(1):
            train_batch_items = []
            if train_idx + batch_size >= len(train_indices):
                np.random.shuffle(train_indices)
                train_idx = 0
            train_batch_items.extend([train_data_loader[train_indices[train_idx + i]] for i in range(batch_size)])
            train_idx += batch_size
            
            # Training step with train data
            loss, predicted, gt = agent.train_step(train_batch_items)
            total_loss += loss
            acc_iterations += 1
            predicted_list.extend(predicted)
            gt_list.extend(gt)

        # Train with aug_train data (aug_times iterations)
        if aug_train_data_loader:
            print(f"Training with aug_train data for {args.aug_times} iterations...")
            for _ in range(args.aug_times):
                aug_batch_items = []
                if aug_train_idx + batch_size >= len(aug_train_indices):
                    np.random.shuffle(aug_train_indices)
                    aug_train_idx = 0
                aug_batch_items.extend([aug_train_data_loader[aug_train_indices[aug_train_idx + i]] for i in range(batch_size)])
                aug_train_idx += batch_size
                
                # Training step with aug_train data
                loss, predicted, gt = agent.train_step(aug_batch_items)
                total_loss += loss
                acc_iterations += 1
                predicted_list.extend(predicted)
                gt_list.extend(gt)

        # Calculate running accuracy
        if len(predicted_list) > 0:
            accuracy = sum([1 for i in range(len(predicted_list)) if predicted_list[i] == gt_list[i]]) / len(predicted_list)
        else:
            accuracy = 0.0

        # Run validation every N iterations
        # print(f"Run validation? {iteration} .. {iteration} % {validation_interval} == 0")
        if (iteration) % validation_interval == 0 and iteration != 0:
            logger.info(f"\nRunning validation at iteration {iteration}...")
            results = run_validation(
                args,
                val_loader,
                agent,
                batch_size=batch_size,
                preload_features=preload_features,
            )

            log_item = {}
            for key, value in results.items():
                log_item[f"val/{key}"] = value
            log_item['val/iteration'] = iteration+1

            log_item['train/loss'] = total_loss / acc_iterations
            log_item['train/accuracy'] = accuracy
            total_loss = 0.0
            acc_iterations = 0
            predicted_list = []
            gt_list = []
            log_item['train/iteration'] = iteration+1

            # Check for best results and save models
            for split_type in results.keys():
                if split_type == "seen":
                    continue
                
                for key in best_results[split_type].keys():
                    # if key != "LE" and results[split_type][key] > best_results[split_type][key]:
                    #     best_results[split_type][key] = results[split_type][key]
                    #     agent.save(iteration, os.path.join(args.model_save_path, f"best_{split_type}_{key}.pth"))
                    #     logger.info(f"New best {split_type}_{key}: {results[split_type][key]}")
                    # elif key == "LE" and results[split_type][key] < best_results[split_type][key]:
                    if key == "LE" and results[split_type][key] < best_results[split_type][key]:
                        best_results[split_type][key] = results[split_type][key]
                        agent.save(iteration, os.path.join(args.model_save_path, f"best_{split_type}_{key}.pth"))
                        logger.info(f"New best {split_type}_{key}: {results[split_type][key]}")
            
            # Check for best average LE (seen + unseen) / 2
            if "seen" in results and "unseen" in results:
                if "LE" in results["seen"] and "LE" in results["unseen"]:
                    current_avg_LE = (results["seen"]["LE"] + results["unseen"]["LE"]) / 2
                    log_item['val/best_avg_LE'] = current_avg_LE
                    if current_avg_LE < best_avg_LE:
                        best_avg_LE = current_avg_LE
                        agent.save(iteration, os.path.join(args.model_save_path, f"best_avg_LE.pth"))
                        logger.info(f"New best average LE: {current_avg_LE:.3f} (seen: {results['seen']['LE']:.3f}, unseen: {results['unseen']['LE']:.3f})")
                       
            # Log to both wandb and file
            if args.wandb_log:
                wandb.log(log_item)
            logger.info(f"Validation results @ iteration {iteration+1}: {log_item}")
                 
            # Save latest model
            agent.save(iteration, os.path.join(args.model_save_path, f"latest.pth"))
            if (iteration - start_iter) % (args.log_every * 5) == 0:
                agent.save(iteration, os.path.join(args.model_save_path, f"cp_{iteration}"))
            
            # Clear GPU cache after validation
            torch.cuda.empty_cache()

            if preload_features:
                _preload_features(train_data_loader, agent, split="seen")
        
        # Clear GPU cache periodically
        if iteration % 50 == 0:
            torch.cuda.empty_cache()
    
    # Final validation
    logger.info(f"\nRunning final validation after {max_iterations} iterations...")
    final_results = run_validation(
        args,
        val_loader,
        agent,
        batch_size=batch_size,
        preload_features=preload_features,
    )
    log_item = {}
    for key, value in final_results.items():
        log_item[f"val/{key}"] = value
    log_item['val/iteration'] = max_iterations
    if args.wandb_log:
        wandb.log(log_item)

    logger.info("Final validation results:")
    logger.info(log_item)
                
    return final_results


def run_validation(args, validation_data_loader, agent, batch_size=4, preload_features=True):
    results = {}
    for split_type in validation_data_loader.keys():
        print(f"Running validation for {split_type}...")
        results[split_type] = valid_loc(
            args, 
            validation_data_loader[split_type], 
            agent, 
            batch_size=batch_size,
            preload_features=preload_features,
            split_type=split_type
        )


         
    return results

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0
    set_random_seed(args.seed + rank)

    # Setup logging first
    logger, log_file = setup_logging(args)
    logger.info(f"Starting training with args: {args}")

    # wandb init
    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            config=args,
            group="gtl",
            id=args.id, 
        )
        logger.info("Wandb logging enabled")
    else:
        logger.info("Wandb logging disabled")

    print("args", args)

    data_paths  = {
        "valseen": args.valseen_data_path,
        "valunseen": args.valunseen_data_path,
        "train": args.train_data_path,
        "aug_train": args.aug_data_paths
    }

    data_loader = {
        "val": {
            "seen": load_data(data_paths["valseen"], load_neighbor=args.use_neighbor_loss, connectivity_dir=args.connectivity_dir, debug=args.debug),
            "unseen": load_data(data_paths["valunseen"], load_neighbor=args.use_neighbor_loss, connectivity_dir=args.connectivity_dir, debug=args.debug)
        },
        "train": load_data(data_paths["train"], load_neighbor=args.use_neighbor_loss, connectivity_dir=args.connectivity_dir, debug=args.debug),
        "aug_train": None
    }

    if args.aug:
        data_loader["aug_train"] = load_data(data_paths["aug_train"], load_neighbor=args.use_neighbor_loss, connectivity_dir=args.connectivity_dir, debug=args.debug)
    # GPU 최적화 옵션 설정
    use_gpu = not getattr(args, 'disable_gpu_features', False)
    preload_all = getattr(args, 'preload_all_features', False)
    
    logger.info(f"Building environment with GPU optimization: {use_gpu}, preload all: {preload_all}")
    env = build_env(
        args.connectivity_dir, 
        args.feat_path, 
        args.image_feat_size, 
        args.batch_size, 
        args.angle_feat_size,
        use_gpu=use_gpu,
        preload_all=preload_all
    )
    
    agent = GraphVlnAgent(args, env)
    # resume file

    start_iter = 0
    if args.resume_file:
        logger.info(f"Resuming from {args.resume_file}")
        start_iter = agent.load(args.resume_file)

    # 로딩 시간을 위해 GPU 메모리 최적화
    preload_features = getattr(args, 'preload_batch_features', True)
    batch_size = args.batch_size

    # New iteration-based training parameters
    max_iterations = getattr(args, 'iterations', 1000)
    validation_interval = getattr(args, 'log_every_iters', 100)

    logger.info(f"Running train_loc with batch_size={batch_size}, max_iterations={max_iterations}, validation_interval={validation_interval}, use_gpu={use_gpu}, preload_features={preload_features}")

    if args.eval_first:
        logger.info("Running initial validation...")
        results = run_validation(
            args,
            data_loader["val"],
            agent,
            batch_size=batch_size,
            preload_features=preload_features,
        )

        log_item = {}
        for key, value in results.items():
            log_item[f"val/{key}"] = value
        log_item['val/iteration'] = start_iter 
        if args.wandb_log:
            wandb.log(log_item)
        logger.info(f"Validation results @ iteration initial (resume from {start_iter}): {log_item}")
        
    if args.train:
        # Use iteration-based training instead of epoch-based
        logger.info("Starting iteration-based training...")
        final_results = train_iteration_based(
            args,
            data_loader["train"],
            data_loader["aug_train"],
            data_loader["val"],
            agent,
            batch_size=batch_size,
            preload_features=preload_features,
            max_iterations=max_iterations,
            validation_interval=validation_interval,
            start_iter=start_iter
        )
        # print(f"Training completed. Final Avg Loss: {avg_loss:.4f}")
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {final_results}")

if __name__ == '__main__':
    main()
