__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import argparse
import logging
import os
import random
import sys
from io import open

import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

# +
from models.modeling_bert import Config, BertForMaskedLM
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.post_training_dataset import BertPostTrainingDataset

from torch.utils.tensorboard import SummaryWriter
# -

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/bert_small_ckpt.bin',
                        type=str,
                        help="checkpoint")
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument("--model_config", default='data/bert_small.json',
                        type=str)

    # Other Parameters
    parser.add_argument("--corpus", default='./rsc/corpus/korquad_corpus_post_training.hdf5', type=str,
                        help="SQuAD corpus for post-training.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Proportion of workers of DataLoader")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    summary_writer = SummaryWriter(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare model
    config = Config.from_json_file(args.model_config)
    model = BertForMaskedLM(config)
    model.bert.load_state_dict(torch.load(args.checkpoint))

    # Multi-GPU Setting
    if n_gpu > 1:
        model = nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)

    post_training_dataset = BertPostTrainingDataset(args.corpus, args.max_seq_length, device)

    num_train_optimization_steps = int(len(post_training_dataset) / args.train_batch_size) * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps * 0.1,
                                     t_total=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(post_training_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    num_train_step = num_train_optimization_steps

    train_sampler = RandomSampler(post_training_dataset)
    train_dataloader = DataLoader(post_training_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    global_step = 0
    epoch = 0

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        iter_bar = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
        tr_step, total_loss, mean_loss = 0, 0., 0.

        for step, batch in enumerate(iter_bar):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)

            loss = model(batch)[0]

            if n_gpu > 1:
                loss = loss.mean()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            tr_step += 1
            total_loss += loss
            mean_loss = total_loss / tr_step
            iter_bar.set_description("Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                     (global_step, num_train_step, mean_loss, loss.item()))
            
            if (global_step + 1) % 100 == 0:
                summary_writer.add_scalar('Train/Mean_Loss', mean_loss, global_step)
                summary_writer.add_scalar('Train/Loss', loss.item(), global_step)

        logger.info("** ** * Saving file * ** **")
        model_checkpoint = "pt_bert_%d.bin" % (epoch)
        logger.info(model_checkpoint)
        output_model_file = os.path.join(args.output_dir, model_checkpoint)
        if n_gpu > 1:
            torch.save(model.module.state_dict(), output_model_file)
        else:
            torch.save(model.state_dict(), output_model_file)
        epoch += 1

if __name__ == '__main__':
    main()
