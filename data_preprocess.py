from torch.utils.data import Dataset, TensorDataset
import logging
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import pickle
from csg_generate import convert_function_to_CSG
import argparse
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import os
import tempfile
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


logger = logging.getLogger(__name__)

class Preprocess(Dataset):
    def __init__(self, tokenizer, args, file_path=''):
        self.train_examples = []
        self.test_examples = []
        self.examples = []
        self.args = args
        self.split_ratio = 0.8
        index_filename = file_path

        # load data from index file (jsonl)
        print("index_filename: ", index_filename)
        url_to_code = {}
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']  # id->func

        data = []
        with open(index_filename.split('.')[0] + '.txt') as f:
            for line in f:
                line = line.strip()
                url, label = line.split('\t')
                if url not in url_to_code:
                    continue
                if label == '0':
                    label = 0
                else:
                    label = 1
                data.append((url, label, tokenizer, args, url_to_code))

        # split data into train and test

        random.shuffle(data)

        train_size = int(len(data) * self.split_ratio)
        train_data = random.sample(data, train_size)
        print("train_data: ", len(train_data))
        test_data = [x for x in data if x not in train_data]
        print("test_data: ", len(test_data))

        for x in tqdm(train_data, total=len(train_data)):
            # Set the signal handler and a 5-second alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                self.train_examples.append(convert_function_to_CSG(x))
            except TimeoutException:
                continue  # If the function call was too long, skip it
            finally:
                signal.alarm(0)  # Cancel the alarm

        with open('preprocessed_data/' + args.dataset_name + '_train.pkl', 'wb') as f:
            pickle.dump(self.train_examples, f)
        print("train_examples: ", len(self.train_examples))

        for x in tqdm(test_data, total=len(test_data)):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                self.test_examples.append(convert_function_to_CSG(x))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

        with open('preprocessed_data/' + args.dataset_name + '_test.pkl', 'wb') as f:
            pickle.dump(self.test_examples, f)
        print("test_examples: ", len(self.test_examples))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:2]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("func: {}".format(url_to_code[example.url]))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("csg_df2code_pos: {}".format(' '.join(map(str, example.csg_df2code_pos))))
                logger.info("csg_df_prenode_id: {}".format(' '.join(map(str, example.csg_df_prenode_id))))
                logger.info("csg_cf2code_pos: {}".format(' '.join(map(str, example.csg_cf2code_pos))))
                logger.info("csg_cf_prenode_id: {}".format(' '.join(map(str, example.csg_cf_prenode_id))))
                logger.info("csg_vul2code_pos: {}".format(' '.join(map(str, example.csg_vul2code_pos))))
                logger.info("csg_vul_prenode_id: {}".format(' '.join(map(str, example.csg_vul_prenode_id))))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # attention mask, represent the edge of CSG
        csg_edge_mask = np.zeros((self.args.code_length + self.args.data_flow_length + self.args.control_flow_length + self.args.vul_relation_length,
                                  self.args.code_length + self.args.data_flow_length + self.args.control_flow_length + self.args.vul_relation_length),
                                 dtype=bool)

        # [cls, code_tokens, sep, csg_data, csg_control, csg_vul, padding]
        code_tokens_length = sum([i > 1 for i in self.examples[item].position_idx])
        no_padding_length = sum([i != 1 for i in self.examples[item].position_idx])

        # code tokens have relation to all code tokens (self-attention)
        csg_edge_mask[:code_tokens_length, :code_tokens_length] = True

        # cls and sep have relations to all code tokens and csg nodes
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                csg_edge_mask[idx, :no_padding_length] = True

        # CSG_df_subgraph corresponds to code tokens
        for idx, (a, b) in enumerate(self.examples[item].csg_df2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length] = True
        # CSG_df_subgraph constructs the data flow between the nodes
        for idx, nodes in enumerate(self.examples[item].csg_df_prenode_id):
            for a in nodes:
                if a + code_tokens_length < len(self.examples[item].position_idx):
                    # child come from father, child: idx+code_tokens_length, father: a + code_tokens_lengths 
                    csg_edge_mask[idx + code_tokens_length, a + code_tokens_length] = True

        # CSG_cf_subgraph corresponds to code tokens
        csg_df_length = len(self.examples[item].csg_df2code_pos)
        for idx, (a, b) in enumerate(self.examples[item].csg_cf2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length + csg_df_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length + csg_df_length] = True
        # CSG_cf_subgraph constructs the control flow between the nodes
        for idx, offset in enumerate(self.examples[item].csg_cf_prenode_id):
            if idx + code_tokens_length + csg_df_length < len(self.examples[item].position_idx):
                # child: idx, father: idx + offset
                csg_edge_mask[idx + code_tokens_length + csg_df_length, idx + code_tokens_length + csg_df_length + offset] = True

        # CSG_vul_subgraph corresponds to code tokens
        csg_cf_length = len(self.examples[item].csg_cf2code_pos)
        for idx, (a, b) in enumerate(self.examples[item].csg_vul2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length + csg_df_length + csg_cf_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length + csg_df_length + csg_cf_length] = True
        # CSG_vul_subgraph constructs the vulnerability relation between the nodes
        for idx, offset in enumerate(self.examples[item].csg_vul_prenode_id):
            if idx + code_tokens_length + csg_df_length + csg_cf_length < len(self.examples[item].position_idx):
                # child: idx, father: idx + offset
                csg_edge_mask[idx + code_tokens_length + csg_df_length + csg_cf_length, code_tokens_length + csg_df_length + csg_cf_length + offset] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(csg_edge_mask),
                torch.tensor(self.examples[item].label))

class Preprocess_JSONL(Dataset):
    def __init__(self, tokenizer, args, file_path=''):
        self.train_examples = []
        self.test_examples = []
        self.examples = []
        self.args = args
        self.split_ratio = 0.8
        index_filename = file_path

        print("index_filename: ", index_filename)
        url_to_code = {}
        data = []
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                if js['label'] == 0:
                    label = 0
                else:
                    label = 1
                data.append((js['index'], label, tokenizer, args, js['code'])) 

        # split data into train and test
        random.shuffle(data)

        train_size = int(len(data) * self.split_ratio)
        train_data = random.sample(data, train_size)
        print("train_data: ", len(train_data))
        test_data = [x for x in data if x not in train_data]
        print("test_data: ", len(test_data))

        for x in tqdm(train_data, total=len(train_data)):
            # Set the signal handler and a 5-second alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                self.train_examples.append(convert_function_to_CSG(x))
            except TimeoutException:
                continue  # If the function call was too long, skip it
            finally:
                signal.alarm(0)  # Cancel the alarm

        with open('preprocessed_data/' + args.dataset_name + '_train.pkl', 'wb') as f:
            pickle.dump(self.train_examples, f)
        print("train_examples: ", len(self.train_examples))

        for x in tqdm(test_data, total=len(test_data)):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                self.test_examples.append(convert_function_to_CSG(x))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

        with open('preprocessed_data/' + args.dataset_name + '_test.pkl', 'wb') as f:
            pickle.dump(self.test_examples, f)
        print("test_examples: ", len(self.test_examples))

        if 'train' in file_path:
            for idx, example in enumerate(self.train_examples[:2]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("func: {}".format(url_to_code[example.url]))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("csg_df2code_pos: {}".format(' '.join(map(str, example.csg_df2code_pos))))
                logger.info("csg_df_prenode_id: {}".format(' '.join(map(str, example.csg_df_prenode_id))))
                logger.info("csg_cf2code_pos: {}".format(' '.join(map(str, example.csg_cf2code_pos))))
                logger.info("csg_cf_prenode_id: {}".format(' '.join(map(str, example.csg_cf_prenode_id))))
                logger.info("csg_vul2code_pos: {}".format(' '.join(map(str, example.csg_vul2code_pos))))
                logger.info("csg_vul_prenode_id: {}".format(' '.join(map(str, example.csg_vul_prenode_id))))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # attention mask, represent the edge of CSG
        csg_edge_mask = np.zeros((self.args.code_length + self.args.data_flow_length + self.args.control_flow_length + self.args.vul_relation_length,
                                  self.args.code_length + self.args.data_flow_length + self.args.control_flow_length + self.args.vul_relation_length),
                                 dtype=bool)

        # [cls, code_tokens, sep, csg_data, csg_control, csg_vul, padding]
        code_tokens_length = sum([i > 1 for i in self.examples[item].position_idx])
        no_padding_length = sum([i != 1 for i in self.examples[item].position_idx])

        # code tokens have relation to all code tokens (self-attention)
        csg_edge_mask[:code_tokens_length, :code_tokens_length] = True

        # cls and sep have relations to all code tokens and csg nodes
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                csg_edge_mask[idx, :no_padding_length] = True

        # CSG_df_subgraph corresponds to code tokens
        for idx, (a, b) in enumerate(self.examples[item].csg_df2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length] = True
        # CSG_df_subgraph constructs the data flow between the nodes
        for idx, nodes in enumerate(self.examples[item].csg_df_prenode_id):
            for a in nodes:
                if a + code_tokens_length < len(self.examples[item].position_idx):
                    # child come from father, child: idx+code_tokens_length, father: a + code_tokens_lengths 
                    csg_edge_mask[idx + code_tokens_length, a + code_tokens_length] = True

        # CSG_cf_subgraph corresponds to code tokens
        csg_df_length = len(self.examples[item].csg_df2code_pos)
        for idx, (a, b) in enumerate(self.examples[item].csg_cf2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length + csg_df_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length + csg_df_length] = True
        # CSG_cf_subgraph constructs the control flow between the nodes
        for idx, offset in enumerate(self.examples[item].csg_cf_prenode_id):
            if idx + code_tokens_length + csg_df_length < len(self.examples[item].position_idx):
                # child: idx, father: idx + offset
                csg_edge_mask[idx + code_tokens_length + csg_df_length, idx + code_tokens_length + csg_df_length + offset] = True

        # CSG_vul_subgraph corresponds to code tokens
        csg_cf_length = len(self.examples[item].csg_cf2code_pos)
        for idx, (a, b) in enumerate(self.examples[item].csg_vul2code_pos):
            if a < code_tokens_length and b < code_tokens_length:
                csg_edge_mask[idx + code_tokens_length + csg_df_length + csg_cf_length, a:b] = True
                csg_edge_mask[a:b, idx + code_tokens_length + csg_df_length + csg_cf_length] = True
        # CSG_vul_subgraph constructs the vulnerability relation between the nodes
        for idx, offset in enumerate(self.examples[item].csg_vul_prenode_id):
            if idx + code_tokens_length + csg_df_length + csg_cf_length < len(self.examples[item].position_idx):
                # child: idx, father: idx + offset
                csg_edge_mask[idx + code_tokens_length + csg_df_length + csg_cf_length, code_tokens_length + csg_df_length + csg_cf_length + offset] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(csg_edge_mask),
                torch.tensor(self.examples[item].label))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':
    print("*********************** Preprocessing ***********************")
    #logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="the dataset txt file path")
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="the dataset name")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--program_language", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=96, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--control_flow_length", default=20, type=int,
                        help="Optional Control Flow input sequence length after tokenization.")
    parser.add_argument("--vul_relation_length", default=12, type=int,
                        help="Optional Vul Relation input sequence length after tokenization.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    Preprocess(tokenizer, args, file_path=args.dataset)
    # Preprocess_JSONL(tokenizer, args, file_path=args.dataset)
    print("*********************** Preprocessing Finish ***********************")
