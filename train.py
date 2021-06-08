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
        wandb.init(project='dkt', config=vars(args))
    
    setSeeds(42) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    valid_data = preprocess.get_valid_data()

    print()
    print(f"# of train_data : {len(train_data)}")
    print(f"# of valid_data : {len(valid_data)}")
    print()

    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)