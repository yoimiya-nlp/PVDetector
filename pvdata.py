from torch.utils.data import Dataset, TensorDataset
import logging
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import pickle

logger = logging.getLogger(__name__)

class PVData(Dataset):
    def __init__(self, tokenizer, args, file_path=''):
        self.examples = []
        self.args = args

        if 'train' in file_path:
            with open('preprocessed_data/' + args.dataset + '_train.pkl', 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in file_path:
            with open('preprocessed_data/' + args.dataset + '_train.pkl', 'rb') as f:
                self.examples = pickle.load(f)
            self.examples = random.sample(self.examples, int(len(self.examples) * 0.125))
        elif 'test' in file_path:
            with open('preprocessed_data/' + args.dataset + '_test.pkl', 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                #logger.info("func: {}".format(url_to_code[example.url]))
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
