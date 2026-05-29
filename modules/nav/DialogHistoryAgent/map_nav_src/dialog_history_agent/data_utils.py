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

# def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True, append_q=False, append_history=False, maximum_navigation_history_length=None):
#     data = []
#     instr_encoding_lengths = []
#     for item in load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test):
#         item['path_id'] = f"{item['instr_id']}"
#         ### instruction
#         instruction = "target : "+item['target']
#         target_only_instruction = "target : "+item['target']
#         if 'a' in item:
#             instruction = f"a: {item['a']} " + instruction
#         if append_q and 'q' in item:
#             instruction = f"q: {item['q']} " + instruction
#         if append_history and item['_chat_idx'] > 1:
#             dialog_history = ''
#             for dialog in item['_full_dialog'][:item['_chat_idx']-1]:
#                 dialog_history = dialog_history + f"q: {dialog['q']} a: {dialog['a']} "
#             instruction = dialog_history + instruction
#         instr_encoding = encode_bert(instruction)
#         target_only_instr_encoding = encode_bert(target_only_instruction)
#         instr_encoding_lengths.append(len(instr_encoding))
#         item['instr_encoding'] = instr_encoding[-max_instr_len:]
#         item['instruction'] = instruction
#         item['target_only_instruction'] = target_only_instruction
#         item['target_only_instr_encoding'] = target_only_instr_encoding


#         # for training with dialog history
#         if maximum_navigation_history_length is not None:
#             if len(item['nav_history']) > maximum_navigation_history_length:
#                 continue
#         if '_full_dialog' in item and 'nav_history' in item:
#             item['context'] = {
#                 '_full_dialog': item['_full_dialog'],
#                 'nav_history': item['nav_history'],
#             }

#         data.append(item)
    
#     metadata = {}
#     metadata['count'] = len(data)
#     metadata['sample'] = random.choice(data)
#     metadata['instr_encoding_length_qurtile'] = np.percentile(instr_encoding_lengths, [25, 50, 75])
#     instr_encoding_lengths = [len(item['instr_encoding']) for item in data]
#     metadata['cut_instr_encoding_length_qurtile'] = np.percentile(instr_encoding_lengths, [25, 50, 75])

#     return data, metadata


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

