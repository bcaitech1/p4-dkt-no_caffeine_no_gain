import os
import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')
    
    parser.add_argument('--train_file_name', default='fixed_train.csv', type=str, help='train file name')
    parser.add_argument('--valid_file_name', default='fixed_valid.csv', type=str, help='valid file name')
    parser.add_argument('--test_file_name', default='fixed_test.csv', type=str, help='test file name')
    
    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')

    parser.add_argument('--model_name', default='', type=str, help='model folder name')
    parser.add_argument('--model_epoch', default=0, type=int, help='epoch')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--output_file', default='output', type=str, help='output directory')
   
    
    parser.add_argument('--max_seq_len', default=20, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    # 모델
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.2, type=float, help='drop out rate')
    parser.add_argument('--dim_div', default=3, type=int, help='model에서 dimension이 커지는 것을 방지')

    # TabNet
    parser.add_argument('--tabnet_pretrain', default=False, type=bool, help='tabnet pretrain')
    parser.add_argument('--use_test_to_train', default=False, type=bool, help='train with testset')
    parser.add_argument('--tabnet_scheduler', default='steplr', type=str, help='tabnet_scheduler')
    parser.add_argument('--tabnet_optimizer', default='adam', type=str, help='tabnet_optimizer')
    parser.add_argument('--tabnet_lr', default=2e-2, type=float, help='tabnet_lr')
    parser.add_argument('--tabnet_batchsize', default=16384, type=int, help='tabnet_batchsize')
    parser.add_argument('--tabnet_n_step', default=10, type=int, help='tabnet_n_step(not log step)')
    parser.add_argument('--tabnet_gamma', default=0.9, type=float, help='tabnet_gamma')
    parser.add_argument('--tabnet_mask_type', default='sparsemax', type=str, help='tabnet_mask_type')
    parser.add_argument('--tabnet_virtual_batchsize', default=256, type=int, help='tabnet_virtual_batchsize')
    parser.add_argument('--tabnet_pretraining_ratio', default=0.8, type=float, help='tabnet_pretraining_ratio')
    
    
    # 훈련
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    parser.add_argument('--is_decoder', default=True, type=bool, help='transformer decoder')

    # Sliding Window
    parser.add_argument('--window', default=False, type=bool, help='Sliding Window augmentation')
    parser.add_argument('--shuffle', default=False, type=bool, help='Shuffle sliding window')
    parser.add_argument('--stride', default=20, type=int, help='Sliding Window stride')
    parser.add_argument('--shuffle_n', default=1, type=int, help='Shuffle times')

    # T-Fixup
    parser.add_argument('--Tfixup', default=False, type=bool, help='Using T-Fixup')
    parser.add_argument('--layer_norm', default=False, type=bool, help='T-Fixup with layer norm')
    
    # log
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')

    # wandb
    parser.add_argument('--use_wandb', default=True, type=bool, help='if you want to use wandb')

    ### 중요 ###
    parser.add_argument('--model', default='lstm', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')
    
    args = parser.parse_args()

    return args