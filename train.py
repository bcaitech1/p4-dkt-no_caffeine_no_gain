import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
def main(args):
    if args.use_wandb:
        wandb.login()
    
    setSeeds(42) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_name)
    preprocess.load_valid_data(args.valid_file_name)
    train_data = preprocess.get_train_data()
    valid_data = preprocess.get_valid_data()
    if args.window:
        train_data = preprocess.sliding_window(train_data, args)
        valid_data = preprocess.sliding_window(valid_data, args)

    if args.use_wandb:
        wandb.init(project='dkt', config=vars(args))
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)