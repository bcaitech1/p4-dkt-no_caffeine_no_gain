import os
import argparse
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
import time
import json

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    
    model_dir = os.path.join(args.model_dir, args.model_name)
    config = json.load(open(f"{model_dir}/exp_config.json", "r"))
    config['model_epoch'] = args.model_epoch
    args = argparse.Namespace(**config)
    
    if args.model == 'tabnet':
        test_data_shift = test_data[test_data['userID'] != test_data['userID'].shift(-1)]
        trainer.tabnet_inference(args, test_data_shift)
    else:
        trainer.inference(args, test_data)
    

if __name__ == "__main__":
    start = time.time()
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    print(f"inference time: {round(time.time() - start)} second")