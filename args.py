import os
import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')
    
    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    
    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model', type=str, help='model file name')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--output_file', default='output', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')
    
    parser.add_argument('--max_seq_len', default=20, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    # 모델
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.2, type=float, help='drop out rate')
    
    # 훈련
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    
    # log
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')

    # wandb
    parser.add_argument('--use_wandb', default=False, type=bool, help='if you want to use wandb')

    ### 중요 ###
    parser.add_argument('--model', default='lstm', type=str, help='model type')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')
    
    args = parser.parse_args()

    return args