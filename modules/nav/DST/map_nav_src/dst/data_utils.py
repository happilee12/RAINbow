import os
import json
import random
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_bert(text):
    return tokenizer.encode(text)

def load_instr_datasets(anno_dir, splits, debug=False):
    data = []
    for split in splits:
        if os.path.exists(os.path.join(anno_dir, '%s.json' % (split))):
            filepath = os.path.join(anno_dir, '%s.json' % (split))
            with open(filepath) as f:
                new_data = json.load(f)
        elif os.path.exists(os.path.join(anno_dir, '%s.jsonl' % (split))):
            filepath = os.path.join(anno_dir, '%s.jsonl' % (split))
            new_data = []
            with open(filepath) as f:
                for line in f:
                    new_data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Invalid annotation file: {anno_dir} {split}")
        
        data += new_data
    if debug:
        return data[:4]
    return data


def construct_instrs(anno_dir, splits, max_action_len=50, teacher_path='gt', instruction_prefix='target : ', debug=False):
    print()
    print(f"Constructing {splits} instrs with gt path as {teacher_path} ({anno_dir})")
    data = []
    errors = {
        'max_action_len': 0,
        'missing_key': 0,
    }
    for item in load_instr_datasets(anno_dir, splits, debug=debug):
        item['path_id'] = f"{item['instr_id']}"
        target_only_instruction = instruction_prefix + item['target']
        target_only_instr_encoding = encode_bert(target_only_instruction)
        item['instruction'] = target_only_instruction
        item['instr_encoding'] = target_only_instr_encoding
        item['target_only_instruction'] = target_only_instruction
        item['target_only_instr_encoding'] = target_only_instr_encoding

        if teacher_path == 'gt': # instance training
            if len(item['nav_history'])+len(item['gt_path']) > max_action_len:
                errors['max_action_len'] += 1
                continue
            item['gt_path'] = item['gt_path']
        elif teacher_path == 'player': # episodic training (from initial vp following player path)
            if len(item['_full_trajectory']) > max_action_len:
                errors['max_action_len'] += 1
                continue
            item['gt_path'] = item['_full_trajectory']
        else:
            raise ValueError(f"Invalid teacher path: {teacher_path}")

        if '_full_dialog' not in item:
            errors['missing_key'] += 1
            continue
        item['dialog'] = item['_full_dialog']
        item['wta'] = [dialog['nav_idx'] for dialog in item['_full_dialog']]

        del item['_full_dialog']
        if '_full_trajectory' in item:
            del item['_full_trajectory']
        if '_start_pano_episode' in item:
            del item['_start_pano_episode']

        data.append(item)
    
    
    print("Data Count: ", len(data))
    print("Data Sample: ", data[0])
    metadata = {}
    metadata['count'] = len(data)
    metadata['sample'] = random.choice(data)

    print(f"Errors: {errors}, loading {anno_dir} {splits} {teacher_path} {max_action_len}")
    return data, metadata

