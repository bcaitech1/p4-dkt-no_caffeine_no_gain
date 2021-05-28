import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
import time
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    

    trainer.inference(args, test_data)
    

if __name__ == "__main__":
    start = time.time()
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    print(f"inference time: {round(time.time() - start)} second")