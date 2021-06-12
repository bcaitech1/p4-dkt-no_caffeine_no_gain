import os
import torch
import numpy as np
import json

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, LastQuery, TfixupBert, Saint, TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from datetime import timedelta, timezone, datetime

import wandb

def tabnet_run(args, train_data, valid_data):
    print(args)
    model_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    json.dump(
        vars(args),
        open(f"{model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"\n{model_dir}/exp_config.json is saved!\n")
    
    
    train_data.pop('userID')
    valid_data.pop('userID')
    
    y_t = train_data['answerCode'].values
    train_data.pop('answerCode')
    x_t = train_data.values
   
    y_v = valid_data['answerCode'].values
    valid_data.pop('answerCode')
    x_v = valid_data.values
    
    print(x_t.dtype)
    
    if args.tabnet_pretrain:
        model = get_tabnet_model(args)
        pre_model, model = model.forward()
        pre_model.fit(
            X_train=train_data,
            eval_set=[valid_data],
            pretraining_ratio=args.tabnet_pretraining_ratio
        )
        pre_model.save_model(f"{model_dir}/pre_model")
        model.fit(
            X_train=x_t, y_train=y_t,
            eval_set=[(x_t, y_t), (x_v, y_v)],
            eval_name=['train', 'valid'],
            max_epochs=args.n_epochs, patience=args.patience,
            eval_metric=['auc', 'accuracy', 'logloss'],
            batch_size=args.tabnet_batchsize, virtual_batch_size=args.tabnet_virtual_batchsize,
            from_unsupervised=pre_model
        )
        model.save_model(f"{model_dir}/model")
        if args.use_wandb:
            for idx in range(len(model.history['train_auc'])):
                wandb.log({
                    'train_auc' : model.history['train_auc'][idx],
                    'train_accuracy' : model.history['train_accuracy'][idx],
                    'train_logloss' : model.history['train_logloss'][idx],
                    'valid_full_auc' : model.history['valid_auc'][idx],
                    'valid_full_accuracy' : model.history['valid_accuracy'][idx],
                    'valid_full_logloss' : model.history['valid_logloss'][idx],
                })
            

    else:
        model = get_tabnet_model(args)
        model = model.forward()
        model.fit(
            X_train=x_t, y_train=y_t,
            eval_set=[(x_t, y_t), (x_v, y_v)],
            eval_name=['train', 'valid'],
            max_epochs=args.n_epochs, patience=args.patience,
            eval_metric=['auc', 'accuracy', 'logloss'],
            batch_size=args.tabnet_batchsize, virtual_batch_size=args.tabnet_virtual_batchsize,
        )
        model.save_model(f"{model_dir}/model")
        if args.use_wandb:
            for idx in range(len(model.history['train_auc'])):
                wandb.log({
                    'train_auc' : model.history['train_auc'][idx],
                    'train_accuracy' : model.history['train_accuracy'][idx],
                    'train_logloss' : model.history['train_logloss'][idx],
                    'valid_full_auc' : model.history['valid_auc'][idx],
                    'valid_full_accuracy' : model.history['valid_accuracy'][idx],
                    'valid_full_logloss' : model.history['valid_logloss'][idx],
                })

            

def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    print(args)
    model_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    json.dump(
        vars(args),
        open(f"{model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    
    print(f"\n{model_dir}/exp_config.json is saved!\n")
            
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc, val_loss = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                       "val_loss": val_loss, "valid_auc": auc, "valid_acc": acc})

        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            model_name = 'model_epoch' + str(epoch) + ".pt"
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                },
                 model_dir, model_name,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)

        targets = input[-4] # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)

        targets = input[-4] # correct

        loss = compute_loss(preds, targets)

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    
    print(f"Valid Loss: {str(loss_avg)}")
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, loss_avg

def tabnet_inference(args, test_data):
    model_dir = os.path.join(args.model_dir, args.model_name)
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(f"{model_dir}/model.zip")
    
    test_data.pop('answerCode')
    test_data.pop('userID')
    loaded_preds = loaded_clf.predict_proba(np.array(test_data.values[:], dtype = np.float64))
    preds = loaded_preds[:, 1]
    
    prediction_name = datetime.now(timezone(timedelta(hours=9))).strftime('%m%d_%H%M')

    output_dir = '/opt/ml/p4-dkt-no_caffeine_no_gain/output/'
    write_path = os.path.join(output_dir, f"{prediction_name}.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))


def inference(args, test_data):
    
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, (args.model_name + "_epoch" + str(args.model_epoch) + ".csv"))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
            


def get_tabnet_model(args):
    if args.tabnet_pretrain:
        pretrain_model,model = TabNet(args)
        pretrain_model.to(args.device)
        model.to(args.device)
        return pretrain_model, model
    else:
        model = TabNet(args)
        model.to(args.device)
        return model


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    elif args.model == 'lstmattn': model = LSTMATTN(args)
    elif args.model == 'bert': model = Bert(args)
    elif args.model == 'lastquery' : model = LastQuery(args)
    elif args.model == 'tfixupbert': model = TfixupBert(args)
    elif args.model == 'saint': model = Saint(args)
    else:
        print("Invalid model!")
        exit()
    
    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):


    features = batch[:-2]
    correct = batch[-2]
    mask = batch[-1]

    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)

    features = [((feature + 1) * mask).to(torch.int64) for feature in features]


    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동
    features = [feature.to(args.device) for feature in features]

    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)


    output = tuple(features + [correct, mask, interaction, gather_index])

    return output



# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss

def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()



def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    
    model_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(model_dir, ('model_epoch' + str(args.model_epoch) + ".pt"))
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model
