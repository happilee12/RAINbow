import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from utils.data import load_nav_graphs
sys.path.append("..")
sys.path.append(".")

class VLNDialogDataset(Dataset):
    """
    Dataset for Vision-Language Navigation with dialog data
    based on the valSeen_data.json format
    """
    def __init__(self, data_path, aug_data_paths = None, tokenizer=None, max_dialog_len=512, connectivity_dir=None, load_neighbor=False, debug=False    ):
        """
        Args:
            data_path: Path to the JSON data file
            tokenizer: Optional tokenizer for processing dialog text
            max_dialog_len: Maximum length of dialog encoding
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_dialog_len = max_dialog_len
        self.connectivity_dir = connectivity_dir
       
        # Load the data
        # self.data = self._load_way_data(load_neighbor=load_neighbor, debug=debug)
        self.data = self._load_rain_data(self.data_path, debug=debug)

        # if self.aug_data_paths is not None:
        #     self.aug_data = self._load_rain_data(self.aug_data_paths, debug=debug)
        #     # self.aug_data = self._load_aug_data(self.aug_data_paths, load_neighbor=load_neighbor, debug=debug)
        #     # simple append two data
        #     self.data.extend(self.aug_data)
        self.scans = list(set([item['scanName'] for item in self.data]))

    def _load_aug_data(self, aug_data_paths, load_neighbor=False, debug=False):
        aug_data = []
        for aug_data_path in aug_data_paths.split(","):
            aug_data.extend(self._load_r2r_data(aug_data_path, load_path=load_neighbor, debug=debug))
        print(f"Added {len(aug_data)} augmented data from {aug_data_paths}")
        return aug_data
    
    def _load_r2r_data(self, data_path, load_path=False, debug=False):
        with open(data_path, 'r') as f:
            data = json.load(f)
        if debug:
            data = data[:4]

        processed_data = []
        for item in data:
            processed_item = {
                'episodeId': item['path_id'],
                'scanName': item['scan'],
                'viewpointId': item['path'][-1],
            }
            if load_path:
                processed_item['neighbor_viewpoint_list'] = item['path'][:-1]
            if self.tokenizer and 'instructions' in item:
                for i in range(len(item['instructions'])):
                    # deep copy processed_item
                    processed_item_copy = processed_item.copy()
                    processed_item_copy['instruction'] = item['instructions'][i]
                    processed_item_copy['instruction_enc'] = self.tokenizer.encode(
                        processed_item_copy['instruction'], 
                        max_length=self.max_dialog_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    ).squeeze(0)

                    processed_data.append(processed_item_copy)
        # print(processed_data[0]["instruction"], processed_data[0]["instruction_enc"])
        return processed_data
    

    def _load_graph(self, scans):
        self.graphs = load_nav_graphs(self.connectivity_dir, scans)
        self.neighbor_viewpoints = {}
        ## get neighbor_viewpoint_list
        for scan in self.graphs:
            self.neighbor_viewpoints[scan] = {}
            for viewpoint in self.graphs[scan]:
                self.neighbor_viewpoints[scan][viewpoint] = list(self.graphs[scan].neighbors(viewpoint))

    def _load_way_data(self, load_neighbor=False, debug=False):
        """Load and preprocess the data from JSON file"""
        print(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        if debug:
            data = data[:4]
        
        # Process the data
        processed_data = []
        for item in data:
            processed_item = {
                'episodeId': item['episodeId'],
                'scanName': item['scanName'],
                'viewpointId': item['finalLocation']['viewPoint'],
                # 'navPath': item['navPath'],
                # 'detailedNavPath': item['detailedNavPath']
            }

            
            # Process dialog if needed
            if self.tokenizer and 'dialogArray' in item:
                # Concatenate dialog turns
                dialog_text = " ".join([item['dialogArray'][1], item['dialogArray'][3]])
                
                # Tokenize the dialog
                dialog_encoding = self.tokenizer.encode(
                    dialog_text, 
                    max_length=self.max_dialog_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                processed_item['instruction'] = dialog_text
                processed_item['instruction_enc'] = dialog_encoding.squeeze(0)
            
 
            processed_data.append(processed_item)
        # print(processed_data[0]["instruction"], processed_data[0]["instruction_enc"])
        

        if load_neighbor:
            scans = list(set([item['scanName'] for item in data]))
            assert self.connectivity_dir is not None, "connectivity_dir is required when load_neighbor is True"
            self._load_graph(scans)
            for item in processed_data:
                item['neighbor_viewpoint_list'] = self.neighbor_viewpoints[item['scanName']][item['viewpointId']]

        print(f"Loaded {len(processed_data)} items from {self.data_path}")
        return processed_data
    
    def _load_rain_data(self, data_path, debug=False):
        """Load and preprocess the data from JSON file"""
        print(f"Loading data from {data_path}")
        # if file is jsonl, load it as jsonl
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                data = [json.loads(line) for line in f]
        else:   
            with open(data_path, 'r') as f:
                data = json.load(f)

        if debug:
            data = data[:4]
        
        # Process the data
        processed_data = []
        for idx, item in enumerate(data):
            if idx % 1000 == 0:
                print(f"Processing {idx} / {len(data)}")
            if 'q' not in item:
                continue
            processed_item = {
                'episodeId': item['instr_id'],
                'scanName': item['scan'],
                'viewpointId': item['start_pano'],
            }

            dialog_text = item["q"]
            dialog_encoding = self.tokenizer.encode(
                dialog_text, 
                max_length=self.max_dialog_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            processed_item['instruction'] = dialog_text
            processed_item['instruction_enc'] = dialog_encoding.squeeze(0)
            
 
            processed_data.append(processed_item)

        print(f"Loaded {len(processed_data)} items from {data_path}")
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_vln_dialog_data(data_path, batch_size=64, tokenizer=None, shuffle=True, num_workers=4):
    """
    Create a DataLoader for VLN dialog data
    
    Args:
        data_path: Path to the JSON data file
        batch_size: Batch size for the DataLoader
        tokenizer: Optional tokenizer for processing dialog text
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for DataLoader
        
    Returns:
        DataLoader object for the dataset
    """
    dataset = VLNDialogDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dialog_collate_fn
    )
    return dataloader

def dialog_collate_fn(batch):
    """
    Custom collate function for dialog data
    
    Args:
        batch: List of data items
        
    Returns:
        Collated batch with tensors and lists
    """
    batch_data = {}
    
    # Extract all keys from the first item
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'dialog_encoding' and 'dialog_encoding' in batch[0]:
            # Stack dialog encodings if they exist
            batch_data[key] = torch.stack([item[key] for item in batch])
        elif key in ['episodeId', 'scanName', 'viewpointId']:
            # Keep these as strings
            batch_data[key] = [item[key] for item in batch]
        else:
            # Default handling
            batch_data[key] = [item[key] for item in batch]
    
    return batch_data

def load_json_data(json_path):
    """
    Simple helper function to load JSON data
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Usage example:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# val_seen_loader = load_vln_dialog_data("path/to/valSeen_data.json", tokenizer=tokenizer)

