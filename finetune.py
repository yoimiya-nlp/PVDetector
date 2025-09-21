# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from pvmodel import PVModel
from pvdata import PVData
from data_preprocess import Preprocess

cpu_cont = 16
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    # Build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)//10
    # args.save_steps = 1
    args.warmup_steps = args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0,0, 0.0
    best_f1 = 0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, position_idx, csg_edge_mask, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss,logits = model(inputs_ids, position_idx, csg_edge_mask, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()

            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = args.to_checkpoint
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        # pretrain = model.encoder
                        # pretrain.save_pretrained("new_pretrain_model")
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.cuda.empty_cache()


def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_dataset = PVData(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval step
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []  
    y_trues = []
    for batch in tqdm(eval_dataloader):
        (inputs_ids, position_idx, csg_edge_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids, position_idx, csg_edge_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    # calculate results
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5

    y_preds = logits[:,1]>best_threshold
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,   
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result


def test(args, model, tokenizer, eval_when_training=False):
    # Build dataloader
    test_dataset = PVData(tokenizer, args, file_path=args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Test step
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(test_dataloader):
        (inputs_ids, position_idx, csg_edge_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(inputs_ids, position_idx, csg_edge_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate results
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,

    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def predict(args, model, tokenizer, best_threshold=0.0):
    # Build dataloader
    pred_dataset = PVData(tokenizer, args, file_path=args.test_data_file)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Predict step
    logger.info("***** Running Predict *****")
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []  
    y_trues = []
    for batch in pred_dataloader:
        (inputs_ids, position_idx, csg_edge_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids, position_idx, csg_edge_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    # Output results
    logits = np.concatenate(logits,0)
    y_preds = logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example,pred in zip(pred_dataset.examples,y_preds):
            if pred:
                f.write(example.url+'\t'+'1'+'\n')
            else:
                f.write(example.url+'\t'+'0'+'\n')
                                                
def main():
    # Setup parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="the dataset name")

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--from_checkpoint", default=None, type=str,
                        help="the checkpoint load from")
    parser.add_argument("--to_checkpoint", default=None, type=str,
                        help="the checkpoint save to")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--control_flow_length", default=64, type=int,
                        help="Optional Control Flow input sequence length after tokenization.")
    parser.add_argument("--vul_relation_length", default=64, type=int,
                        help="Optional Vul Relation input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run pred on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)

    # Setup seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # pvdetector_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    pvdetector_model = RobertaForSequenceClassification(config)

    pvdetector_model = PVModel(pvdetector_model,config,tokenizer,args)
    total_num = sum(p.numel() for p in pvdetector_model.parameters())
    # print("Total parameters: ", total_num)
    logger.info("Training/evaluation parameters %s", args)

    # Train Phrase
    if args.do_train:
        train_dataset = PVData(tokenizer, args, file_path=args.train_data_file)
        # continue training
        #checkpoint_prefix = args.from_checkpoint + '/model.bin'
        #output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        #pvdetector_model.load_state_dict(torch.load(output_dir))
        #pvdetector_model.to(args.device)
        train(args, train_dataset, pvdetector_model, tokenizer)

    # Test Phrase
    results = {}
    if args.do_test:
        checkpoint_prefix = args.from_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        pvdetector_model.load_state_dict(torch.load(output_dir))
        pvdetector_model.to(args.device)
        result = test(args, pvdetector_model, tokenizer)
        print("Test result: ", result)
        
    if args.do_predict:
        checkpoint_prefix = args.from_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        pvdetector_model.load_state_dict(torch.load(output_dir))
        pvdetector_model.to(args.device)
        predict(args, pvdetector_model, tokenizer, best_threshold=0.5)
        print("Predict over")

    return results


if __name__ == "__main__":
    main()
