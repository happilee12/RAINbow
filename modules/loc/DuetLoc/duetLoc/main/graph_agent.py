import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
# from .eval_utils import cal_dtw

from duetLoc.models.graph_utils_for_loc import GraphMap
from duetLoc.models.model import VLNBert, Critic
from duetLoc.models.ops import pad_tensors_wgrad


class GraphVlnAgent(Seq2SeqAgent):
 
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}
        self.graph_enc_cache = {}
        # panorama feature cache
        self.pano_feat_cache = {}
        self.tokenizer = self.load_tokenizer()

    def _language_variable(self, encoded_instructions):
        # encode 
        seq_lengths = [len(ei) for ei in encoded_instructions]
        seq_tensor = np.zeros((len(encoded_instructions), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(encoded_instructions), max(seq_lengths)), dtype=np.bool)
        for i, ei in enumerate(encoded_instructions):
            seq_tensor[i, :seq_lengths[i]] = ei
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable with GPU caching. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        cache_miss_count = 0
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            scan_vp = f"{ob['scan']}_{ob['viewpoint']}"
            
            # Check if features are in cache
            if scan_vp in self.pano_feat_cache:
                cached_feats = self.pano_feat_cache[scan_vp]
                view_img_fts = cached_feats['view_img_fts']
                view_ang_fts = cached_feats['view_ang_fts']
                nav_types = cached_feats['nav_types']
                cand_vpids = cached_feats['cand_vpids']
            else:
                # print("panorama feature cache miss", scan_vp)
                cache_miss_count += 1

                # cand views
                used_viewidxs = set()
                for j, cc in enumerate(ob['candidate']):
                    view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                    view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                    nav_types.append(1)
                    cand_vpids.append(cc['viewpointId'])
                    used_viewidxs.add(cc['pointId'])
                
                # non cand views
                view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                    in enumerate(ob['feature']) if k not in used_viewidxs])
                view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                    in enumerate(ob['feature']) if k not in used_viewidxs])
                nav_types.extend([0] * (36 - len(used_viewidxs)))
                
                # Convert to tensors and move to GPU
                view_img_fts = torch.stack(view_img_fts, 0).cuda()    # (n_views, dim_ft)
                view_ang_fts = torch.stack(view_ang_fts, 0).cuda()
                
                # Create location features on GPU
                view_box_fts = torch.ones((len(view_img_fts), 3), dtype=torch.float32, device='cuda')
                view_loc_fts = torch.cat([view_ang_fts, view_box_fts], 1)
                
                # Cache the features on GPU
                self.pano_feat_cache[scan_vp] = {
                    'view_img_fts': view_img_fts,
                    'view_ang_fts': view_ang_fts,
                    'nav_types': nav_types,
                    'cand_vpids': cand_vpids,
                    'view_loc_fts': view_loc_fts
                }
            
            # Use cached features
            batch_view_img_fts.append(view_img_fts)
            batch_loc_fts.append(self.pano_feat_cache[scan_vp]['view_loc_fts'])
            batch_nav_types.append(torch.LongTensor(nav_types).cuda())
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts)
        batch_loc_fts = pad_tensors(batch_loc_fts)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0)
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        # print(f"Cache miss count: {cache_miss_count}")

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _get_pano_features(self, scan, vp):
        ''' Extract precomputed features into variable with GPU caching. '''
        
        view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
        scan_vp = f"{scan}_{vp}"
        
        # Check if features are in cache
        if scan_vp in self.pano_feat_cache:
            print("panorama feature cache hit", scan_vp)
            # cached_feats = self.pano_feat_cache[scan_vp]
            # view_img_fts = cached_feats['view_img_fts']
            # view_ang_fts = cached_feats['view_ang_fts']
            # nav_types = cached_feats['nav_types']
            # cand_vpids = cached_feats['cand_vpids']
        else:
            # cand views
            used_viewidxs = set()
            self.env.env.newEpisodes([scan], [vp], [3.14])
            ob = self.env.env.sims[0]
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            
            # Convert to tensors and move to GPU
            view_img_fts = torch.stack(view_img_fts, 0).cuda()    # (n_views, dim_ft)
            view_ang_fts = torch.stack(view_ang_fts, 0).cuda()
            
            # Create location features on GPU
            view_box_fts = torch.ones((len(view_img_fts), 3), dtype=torch.float32, device='cuda')
            view_loc_fts = torch.cat([view_ang_fts, view_box_fts], 1)
            
            # Cache the features on GPU
            self.pano_feat_cache[scan_vp] = {
                'view_img_fts': view_img_fts,
                'view_ang_fts': view_ang_fts,
                'nav_types': nav_types,
                'cand_vpids': cand_vpids,
                'view_loc_fts': view_loc_fts
            }
        
        return self.pano_feat_cache[scan_vp]

    def _loc_gmap_variable(self, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(gmaps)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids = list(gmap.node_positions.keys())
            # visited_vpids, unvisited_vpids = [], []                
            # for k in gmap.node_positions.keys():
            #     if self.args.act_visited_nodes:
            #         if k == obs[i]['viewpoint']:
            #             visited_vpids.append(k)
            #         else:
            #             unvisited_vpids.append(k)
            #     else:
            #         if gmap.graph.visited(k):
            #             visited_vpids.append(k)
            #         else:
            #             unvisited_vpids.append(k)
            batch_no_vp_left.append(True)
            gmap_vpids = visited_vpids
            gmap_visited_masks = [0] * len(gmap_vpids)


            #     gmap_vpids = [None] + visited_vpids + unvisited_vpids
            #     gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            # else:
            #     gmap_vpids = [None] + unvisited_vpids
            #     gmap_visited_masks = [0] * len(gmap_vpids)

            
            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            # gmap_pos_fts = gmap.get_pos_fts(
            #     obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            # )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            # batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        # batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 
            # 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def test(self, batch_items, cache_graph_enc=False, save_gmap_path=None):
        
        batch_scans = [item['scanName'] for item in batch_items]
        batch_viewpoints = [item['viewpointId'] for item in batch_items]
        batch_encocded_instructions = [item['instruction_enc'] for item in batch_items]
        
        language_inputs = self._language_variable(batch_encocded_instructions)
        txt_embeds = self.vln_bert('language', language_inputs)
        
        
        # Prepare navigation inputs for entire batch
        gmaps = self.prepare_gmaps(batch_items) 
        nav_inputs = self._loc_gmap_variable(gmaps)
        nav_inputs.update({
            'txt_embeds': txt_embeds,
            'txt_masks': language_inputs['txt_masks'],
        })

        # Forward pass for entire batch
        loc_outs = self.vln_bert('localize', nav_inputs)
        logits = loc_outs['global_logits']
        
        # Get predictions
        _, max_idx = torch.max(logits, dim=1)
        predicted = []
        for i, gmap in enumerate(gmaps):
            nodes = list(gmap.node_positions.keys())
            predicted.append(nodes[max_idx[i]])
        
        return predicted, batch_viewpoints
    
    def _random_viewpoint_in_graph(self, scanId):
        return random.choice(list(self.env.shortest_paths[scanId].keys()))
        
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        cfg_name = 'bert-base-uncased'
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                tokenizer = AutoTokenizer.from_pretrained(cfg_name)
                return tokenizer
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
    
    def localize(self, texts, scanIds):
    
        # self.env._load_nav_graphs(scanIds)

        batch_items = []
        for i, scanId in enumerate(scanIds):
            # text = texts[i]
            # text_enc = tokenizer.encode(text, max_length=self.args.max_dialog_len, padding='max_length', truncation=True)
            batch_items.append({
                'scanName': scanId,
                'viewpointId': self._random_viewpoint_in_graph(scanId),
                # 'instruction_enc': text_enc,
            })
        # print("Localize batch_items .. ", batch_items)
        
        encoded_instructions = [
            self.tokenizer.encode(text, max_length=self.args.max_dialog_len, padding='max_length', truncation=True)
            for text in texts
        ]
        language_inputs = self._language_variable(encoded_instructions)
        txt_embeds = self.vln_bert('language', language_inputs)

        # Prepare navigation inputs for entire batch
        gmaps = self.prepare_gmaps(batch_items) 
        nav_inputs = self._loc_gmap_variable(gmaps)
        nav_inputs.update({
            'txt_embeds': txt_embeds,
            'txt_masks': language_inputs['txt_masks'],
        })

        # Forward pass for entire batch
        loc_outs = self.vln_bert('localize', nav_inputs)
        logits = loc_outs['global_logits']
        
        # Get predictions
        _, max_idx = torch.max(logits, dim=1)
        predicted = []
        for i, gmap in enumerate(gmaps):
            nodes = list(gmap.node_positions.keys())
            predicted.append(nodes[max_idx[i]])
        
        return predicted


    def calculate_loss(self, logits, gmaps, batch_items):
        batch_viewpoints = [item['viewpointId'] for item in batch_items]

        if self.args.use_neighbor_loss:
            batch_neighbor_viewpoints = [item['neighbor_viewpoint_list'] for item in batch_items]
            
            # 수치적 안정성을 위한 epsilon 추가
            epsilon = 1e-8
            
            # log_softmax 대신 softmax 사용하고 log 취하기
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + epsilon)
            
            target_distributions = []
            for i, gmap in enumerate(gmaps):
                nodes = list(gmap.node_positions.keys())
                gt_vp = batch_viewpoints[i]
                neighbor_vps = batch_neighbor_viewpoints[i]
                num_nodes = len(nodes)
                distribution = torch.zeros(num_nodes, device=logits.device)

                # 정답 노드 인덱스
                try:
                    gt_idx = nodes.index(gt_vp)
                except ValueError:
                    raise ValueError(f"gt_vp {gt_vp} not in nodes {nodes}")
                    # gt_idx = None

                # neighbor 노드 인덱스
                neighbor_indices = []
                for n_vp in neighbor_vps:
                    try:
                        idx = nodes.index(n_vp)
                        neighbor_indices.append(idx)
                    except ValueError:
                        raise ValueError(f"neighbor_vp {n_vp} not in nodes {nodes}")
                        # continue

                # 중복 제거 (정답 노드가 neighbor에 포함될 수 있음)
                neighbor_indices = list(set(neighbor_indices) - {gt_idx})

                neighbor_prob = self.args.neighbor_loss_weight
                gt_prob = 1 - self.args.neighbor_loss_weight
                distribution[gt_idx] = gt_prob
                if neighbor_indices:
                    prob_per_neighbor = neighbor_prob / len(neighbor_indices)
                    for n_idx in neighbor_indices:
                        distribution[n_idx] = prob_per_neighbor

                # 정규화
                distribution = distribution / distribution.sum()
                target_distributions.append(distribution)

            # print("target_distributions .. ", target_distributions, log_probs)
            max_len = max(t.size(0) for t in target_distributions)

            # 각 텐서를 패딩하여 같은 크기로 만듭니다
            padded_distributions = []
            for t in target_distributions:
                if t.size(0) < max_len:
                    padding = torch.zeros(max_len - t.size(0), device=t.device)
                    padded_t = torch.cat([t, padding])
                    padded_distributions.append(padded_t)
                else:
                    padded_distributions.append(t)

            # 이제 같은 크기의 텐서들을 스택할 수 있습니다
            target_distributions = torch.stack(padded_distributions, 0)
            
            # target_distributions에 epsilon 추가하여 0 방지
            target_distributions = target_distributions + epsilon
            target_distributions = target_distributions / target_distributions.sum(dim=-1, keepdim=True)
            
            # 값의 범위 제한
            target_distributions = torch.clamp(target_distributions, min=epsilon, max=1-epsilon)
            log_probs = torch.clamp(log_probs, min=-100, max=100)  # log 값의 범위 제한
            
            # KL divergence 계산
            loss = torch.nn.functional.kl_div(
                log_probs, 
                target_distributions, 
                reduction='batchmean',
                log_target=False
            )
            

        else:
            labels = []
            valid_mask = []
            for i, gmap in enumerate(gmaps):
                nodes = list(gmap.node_positions.keys())
                gt_vp = batch_viewpoints[i]
                try:
                    label_idx = nodes.index(gt_vp)
                    labels.append(label_idx)
                    valid_mask.append(True)
                except ValueError:
                    # Skip invalid samples by adding a dummy label and marking as invalid
                    labels.append(0)  # dummy label
                    valid_mask.append(False)
            
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=logits.device)

            # Only compute loss for valid samples
            if valid_mask.any():
                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]
                loss = self.criterion(valid_logits, valid_labels)
            else:
                # If no valid samples, return zero loss
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss

    def prepare_gmaps(self, batch_items):
        batch_scans = [item['scanName'] for item in batch_items]
        batch_viewpoints = [item['viewpointId'] for item in batch_items]
        gmaps = [GraphMap(vp) for vp in batch_viewpoints]
        gmaps = self.build_all_map_embeddings(batch_scans, gmaps, batch_size=1024)
        return gmaps

    def train_step(self, batch_items):
        batch_scans = [item['scanName'] for item in batch_items]
        batch_encocded_instructions = [item['instruction_enc'] for item in batch_items]
        batch_viewpoints = [item['viewpointId'] for item in batch_items]
        language_inputs = self._language_variable(batch_encocded_instructions)
        txt_embeds = self.vln_bert('language', language_inputs)
        gmaps = self.prepare_gmaps(batch_items)
        nav_inputs = self._loc_gmap_variable(gmaps)
        nav_inputs.update({
            'txt_embeds': txt_embeds,
            'txt_masks': language_inputs['txt_masks'],
        })
        
        # Forward pass for entire batch
        loc_outs = self.vln_bert('localize', nav_inputs)
        logits = loc_outs['global_logits']
        
        # Calculate loss
        loss = self.calculate_loss(logits, gmaps, batch_items)
        
        # Backward and optimize
        self.vln_bert_optimizer.zero_grad()
        loss.backward()
        self.vln_bert_optimizer.step()
        
        # Get predictions
        _, max_idx = torch.max(logits, dim=1)
        predicted = []
        for i, gmap in enumerate(gmaps):
            nodes = list(gmap.node_positions.keys())
            pred_vp = nodes[max_idx[i]]
            predicted.append(pred_vp)
        
        return loss.item(), predicted, batch_viewpoints

    def clear_pano_cache(self):
        """Clear the panorama feature cache to free GPU memory"""
        self.pano_feat_cache = {}
        torch.cuda.empty_cache()

    def build_map_embeddings(self, scan_id, gmap: GraphMap, batch_size):
        """
        Build embeddings for all viewpoints in a given scan.
        
        Args:
            scan_id (str): The scan ID to process
            batch_size (int): Batch size for processing panorama features
            
        Returns:
            dict: Dictionary mapping viewpoint IDs to their embeddings
        """
        import time
        
        # 1. Get all viewpoints for the scan
        start_time = time.time()
        all_viewpoints = list(self.env.graphs[scan_id].nodes.keys())
        # print(f"Processing {len(all_viewpoints)} viewpoints for scan {scan_id}")
        # print(f"Time to get viewpoints: {time.time() - start_time:.2f}s")
        
        # 2. Process viewpoints in batches
        viewpoint_embeddings = {}
        total_batches = (len(all_viewpoints) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_viewpoints), batch_size):
            batch_start_time = time.time()
            # print(f"\nProcessing batch {i//batch_size + 1}/{total_batches}")
            
            # Initialize environment
            # init_start = time.time()
            batch_vps = all_viewpoints[i:i + batch_size]
            self.env.env._init_sims(len(batch_vps))
            # print(f"Time to initialize environment: {time.time() - init_start:.2f}s")
            
            # Get panorama features
            # pano_start = time.time()
            self.env.env.newEpisodes([scan_id] * len(batch_vps), batch_vps, [3.14] * len(batch_vps))
            obs_batch = self.env._get_obs()
            # print(f"Time to get panorama features: {time.time() - pano_start:.2f}s")
            
            # Update graph in batch
            # graph_start = time.time()
            gmap.update_graph_batch(obs_batch, scan_id)
            # print(f"Time to update graph: {time.time() - graph_start:.2f}s")

            # Get panorama features and embeddings
            # embed_start = time.time()
            pano_inputs = self._panorama_feature_variable(obs_batch)
            # print(f"Time to prepare panorama inputs: {time.time() - embed_start:.2f}s")

            # Get embeddings from model
            # model_start = time.time()
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            # print(f"Time to get model embeddings: {time.time() - model_start:.2f}s")
            
            # Calculate average embeddings
            # avg_start = time.time()
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                            torch.sum(pano_masks, 1, keepdim=True)
            # print(f"Time to calculate average embeddings: {time.time() - avg_start:.2f}s")
            
            # Store embeddings
            # store_start = time.time()
            for j, vp in enumerate(batch_vps):
                viewpoint_embeddings[vp] = avg_pano_embeds[j]
            # print(f"Time to store embeddings: {time.time() - store_start:.2f}s")
            
            # Update node embeddings
            # update_start = time.time()
            for vp_id, embed in viewpoint_embeddings.items():
                gmap.update_node_embed(vp_id, embed, rewrite=True)
            # print(f"Time to update node embeddings: {time.time() - update_start:.2f}s")
            
            # Clear GPU cache
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
            
            # print(f"Total time for batch: {time.time() - batch_start_time:.2f}s")
        
        return gmap

    def build_all_map_embeddings(self, scan_ids, gmaps: List[GraphMap], batch_size) -> List[GraphMap]:
        """
        Build embeddings for all viewpoints in multiple scans.
        
        Args:
            scan_ids (list): List of scan IDs to process
            gmaps (List[GraphMap]): List of graph maps to process
            batch_size (int): Batch size for processing panorama features
            
        Returns:
            Dict[str, GraphMap]: Dictionary mapping scan IDs to GraphMap objects
        """
        gmap_with_embeddings = []
        for idx, scan_id in enumerate(scan_ids):
            gmap = self.build_map_embeddings(scan_id, gmaps[idx], batch_size)
            gmap_with_embeddings.append(gmap)
            
        return gmap_with_embeddings