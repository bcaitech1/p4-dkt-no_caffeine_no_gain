import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
from dkt.trainer import update_train_data
import torch
from dkt.utils import setSeeds
import wandb
import json
import argparse

def main(args):
    if args.use_wandb:
        wandb.login()
        wandb.init(project='dkt', config=vars(args))
    
    setSeeds(42) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.original_data:
        args.train_file_name = "original_fixed_train.csv"
        args.valid_file_name = "original_fixed_valid.csv"
        args.test_file_name = "test_data_add_elapsed.csv"

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_name,args.valid_file_name)
    preprocess.load_valid_data(args.valid_file_name)
    train_data = preprocess.get_train_data()
    valid_data = preprocess.get_valid_data()
    test_data = None


    if args.use_pseudo:
        preprocess.load_test_data(args.test_file_name)
        test_data = preprocess.get_test_data()
    
    if args.model == 'tabnet':
        trainer.tabnet_run(args, train_data, valid_data, test_data)
    else:
        trainer.run(args, train_data, valid_data, test_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
