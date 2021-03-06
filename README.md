# π No Caffeine No Gain

<br>

**νλ‘μ νΈ κΈ°κ° : 2021.05.24 ~ 2021.06.15**
<br>

**νλ‘μ νΈ μ£Όμ  : Deep Knowledge Tracing**
<br>
<br>

## [λͺ©μ°¨]

- [\[Deep Knowledge Tracing μκ°\]](#deep-knowledge-tracing-μκ°)
- [[Installation]](#installation)
  * [Dependencies](#dependencies)
- [[Usage]](#usage)
  * [Dataset](#dataset)
  * [Train](#train)
  * [Inference](#inference)
  * [Arguments](#arguments)
- [[File Structure]](#file-structure)
  * [LSTM](#lstm)
  * [LSTMATTN](#lstmattn)
  * [BERT](#bert)
  * [LGBM](#lgbm)
  * [SAINT](#saint)
  * [LastQuery](#lastquery)
  * [TABNET](#tabnet)
- [[Input CSV File]](#input-csv-file)
- [[Feature]](#feature)
- [[Contributors]](#contributors)
- [[Collaborative Works]](#collaborative-works)
  * [π Notion](#-notion)
- [[Reference]](#reference)
  * [Papers](#papers)
  * [Dataset](#dataset-1)


<br>
<br>

## [Deep Knowledge Tracing μκ°]

**DKT**λ **Deep Knowledge Tracing**μ μ½μλ‘ μ°λ¦¬μ "μ§μ μν"λ₯Ό μΆμ νλ λ₯λ¬λ λ°©λ²λ‘ μλλ€.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122540925-10143980-d064-11eb-8afc-ccdb1e76114c.png' height='250px '/>
</div>
<br/>


λνμμλ νμ κ°κ°μΈμ μ΄ν΄λλ₯Ό κ°λ¦¬ν€λ μ§μ μνλ₯Ό μμΈ‘νλ μΌλ³΄λ€λ, μ£Όμ΄μ§ λ¬Έμ λ₯Ό λ§μΆμ§ νλ¦΄μ§ μμΈ‘νλ κ²μ μ§μ€ν©λλ€.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541124-42be3200-d064-11eb-8e60-0f7d82a15df9.png' height='250px '/>
</div>
<br/>

<br>

## [Installation]

### Dependencies

- torch
- pandas
- sklearn
- pycaret
- tqdm
- wandb
- easydict
- pytorch-tabnet

```bash
pip install -r requirements.txt
```

<br>
<br>

## [Usage]

### Dataset

νμ΅μ νμν λ°μ΄ν°λ₯Ό λ§λ€κΈ° μν΄ λ κ°μ `.py` νμΌμ μμ°¨μ μΌλ‘ μ€νν΄μΌ ν©λλ€.

```bash
$ p4-dkt-no_caffeine_no_gain# python make_elapsed.py
$ p4-dkt-no_caffeine_no_gain# python make_fixed_data.py
```

<br>

### Train

λͺ¨λΈμ νμ΅νκΈ° μν΄μλ `train.py` λ₯Ό μ€νμν΅λλ€.

μλ Arguments μ μλ argument μ€ νμν argumet λ₯Ό λ°κΏ μ¬μ©νλ©΄ λ©λλ€.

```bash
$ p4-dkt-no_caffeine_no_gain# python train.py
```

μ΄ 7κ°μ§μ λͺ¨λΈμ μ νν  μ μμ΅λλ€.

- **TABNET**
- **LASTQUERY**
- **SAINT**
- **LGBM**
- **BERT**
- **LSTMATTN**
- **LSTM**

<br>

### Inference

νμ΅λ λͺ¨λΈλ‘ μΆλ‘ νκΈ° μν΄μλ `inference.py` λ₯Ό μ€νμν΅λλ€.

νμν argument λ `β-model_name` κ³Ό `β-model_epoch` μλλ€.

```bash
$ p4-dkt-no_caffeine_no_gain# python inference.py --model_name "νμ΅ν λͺ¨λΈ ν΄λ μ΄λ¦" --model_epoch "μ¬μ©νκ³ ν λͺ¨λΈμ epoch"
```

<br>

### Arguments

train κ³Ό inference μμ νμν argument μλλ€.

```plain-text
# Basic
--model: model type (default:'lstm')
--scheduler: scheduler type (default:'plateau')
--device: device to use (defualt:'cpu')
--data_dir: data directory (default:'/opt/ml/input/data/train_dataset')
--asset_dir: asset directory (default:'asset/')
--train_file_name: train file name (default:'add_FE_fixed_train.csv')
--valid_file_name: validation file name (default:'add_FE_fixed_valid.csv')
--test_file_name: test file name (default:'add_FE_fixed_test.csv')
--model_dir: model directory (default:'models/')
--num_workers: number of workers (default:1)
--output_dir: output directory (default:'output/')
--output_file: output file name (default:'output')
--model_name: model folder name (default:'')
--model_epoch: model epoch to use (default:1)

# Hyperparameters
--seed: random state (default:42)
--optimizer: optimizer type (default:'adamW')
--max_seq_len: max sequence length (default:20)
--hidden_dim: hidden dimension size (default:64)
--n_layers: number of layers (default:2)
--n_epochs: number of epochs (default:20)
--batch_size: batch size (default:64)
--lr: learning rate (default:1e-4)
--clip_grad: clip grad (default:10)
--patience: for early stopping (default:5)
--drop_out: drop out rate (default:0.2)
--dim_div: hidden dimension dividor in model to prevent too be large scale (default:3)

# Transformer
--n_heads: number of heads (default:2)
--is_decoder: use transformer decoder (default:True)

# TabNet
--tabnet_pretrain: Using TabNet pretrain (default:False)
--use_test_to_train: to training includes test data (default:False)
--tabnet_scheduler: TabNet scheduler (default:'steplr')
--tabnet_optimizer: TabNet optimizer (default:'adam')
--tabnet_lr: TabNet learning rate (default:2e-2)
--tabnet_batchsize: TabNet batchsize (default:16384)
--tabnet_n_step: TabNet n step(not log step) (default:5)
--tabnet_gamma: TabNet gamma (default:1.7)
--tabnet_mask_type: TabNet mask type (default:'saprsemax')
--tabnet_virtual_batchsize: TabNet virtual batchsize (default:256)
--tabnet_pretraining_ratio: TabNet pretraining ratio (default:0.8)

# Sliding Window
--window: Using Sliding Window augmentation (default:False)
--shuffle: shuffle Sliding Window (default:False)
--stride: Sliding Window stride (default:20)
--shuffle_n: Shuffle times (default:1)

# T-Fixup
--Tfixup: Using T-Fixup (default:False)
--layer_norm: T-Fixup with layer norm (default:False)

# Pseudo Labeling
--use_pseudo: Using Pseudo Labeling (default:False)
--pseudo_label_file: file path for Pseudo Labeling (default:'')

# log
--log_steps: print log per n steps (default:50)

# wandb
--use_wandb: if you want to use wandb (default:True)
```

<br>
<br>

## [File Structure]

μ μ²΄μ μΈ File Structure μλλ€.

```
code
βββ README.md
βββ .gitignore
βββ args.py
βββ make_custom_data
β   βββ make_elapsed.py - time κ΄λ ¨ feature μμ±
β   βββ make_fixed_data.py - user μ λ΅λ₯  κΈ°λ°μΌλ‘ valid μμ±
β   βββ make_original_fixed_data.py - shuffleν΄μ valid μμ±
β
βββ dkt
β   βββ criterion.py
β   βββ dataloader.py
β   βββ metric.py
β   βββ model.py
β   βββ optimizer.py
β   βββ scheduler.py
β   βββ trainer.py
β   βββ utils.py
βββ ensemble.py
βββ inference.py
βββ requirements.txt - dependencies
βββ train.py

```

<br>
<br>

### LSTM

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541256-66817800-d064-11eb-92ea-7fc9ae8b0cce.png' height='250px '/>
</div>
<br/>

- sequence dataλ₯Ό λ€λ£¨κΈ° μν LSTM λͺ¨λΈμλλ€.
- **κ΅¬ν**

    ```
    model.py
    βββ class LSTM
    β   βββ init()
    βββ βββ forward() : return predicts

    args.py
    βββ args.max_seq_len(default : 20)
    βββ args.n_layers(default : 2)
    βββ args.n_heads(default : 2)
    βββ args.hidden_dim(default : 64)
    ```

<br>
<br>

### LSTMATTN

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541497-a6485f80-d064-11eb-8c97-61b7b9d25954.png' height='250px '/>
</div>
<br/>


- LSTM λͺ¨λΈμ Self-Attentionμ μΆκ°ν λͺ¨λΈμλλ€.
- **κ΅¬ν**

    ```
    model.py
    βββ class LSTMATTN
    β   βββ init()
    βββ βββ forward() : return predicts

    args.py
    βββ args.max_seq_len(default : 20)
    βββ args.n_layers(default : 2)
    βββ args.n_heads(default : 2)
    βββ args.hidden_dim(default : 64)
    ```

<br>
<br>

### BERT

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541560-b6f8d580-d064-11eb-94ac-73c0acafc796.png' height='250px '/>
</div>
<br/>


- `Huggingface` μμ BERT κ΅¬μ‘°λ₯Ό κ°μ Έμμ μ¬μ©ν©λλ€. λ€λ§, pre-trained λͺ¨λΈμ΄ μλκΈ° λλ¬Έμ Transformer-encoder μ κ°μ΅λλ€.
- νμ¬ λͺ¨λΈμμλ bert_config μ is_decoder λ₯Ό True λ‘ μ£Όμ΄ Transformer-decoder λ‘ μ¬μ©νκ³  μμ΅λλ€.
- **κ΅¬ν**

    ```
    model.py
    βββ class Bert
    β   βββ init()
    βββ βββ forward() : return predicts

    args.py
    βββ args.max_seq_len(default : 20)
    βββ args.n_layers(default : 2)
    βββ args.n_heads(default : 2)
    βββ args.is_decoder(default : True)
    βββ args.hidden_dim(default : 64)
    ```

<br>
<br>

### LGBM

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541790-03dcac00-d065-11eb-9464-3f4c890bccda.png' height='200px '/>
</div>
<br/>


- tabular dataμμ μ’μ μ±λ₯μ λ³΄μ΄λ Machine Learning λͺ¨λΈμλλ€.
- **κ΅¬ν**

    ```
    model.py
    βββ class LGBM
    β   βββ init()
    βββ βββ fit() : return trained model
    ```

<br>
<br>

### SAINT

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541828-0d661400-d065-11eb-9028-d7b1a0d6adce.png' height='250px '/>
</div>
<br/>



- Kaggle Riiid AIEd Challenge 2020μ [Hostκ° μ μν solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/193250) μλλ€.
- Transformerμ λΉμ·ν κ΅¬μ‘°μ λͺ¨λΈλ‘ Encoderμ Decoderλ₯Ό κ°μ§κ³  μμ΅λλ€.
- μΈμ½λλ feature μλ² λ© μ€νΈλ¦Όμ self-attention λ μ΄μ΄λ₯Ό μ μ©νκ³  λμ½λμμ self-attention λ μ΄μ΄μ μΈμ½λ-λμ½λ attention λ μ΄μ΄λ₯Ό μλ΅ μλ² λ©κ³Ό μΈμ½λμ μΆλ ₯ μ€νΈλ¦Όμ λ²κ°μ μ μ©νλ κ΅¬μ‘°μλλ€.
- **Paper Review** : [[Saint λͺ¨λΈ λΆμ]](https://www.notion.so/Saint-507d13692825492ba05128f4548c2da7)
- **κ΅¬ν**

    ```
    model.py
    βββ class Saint
    β   βββ init()
    βββ βββ forward() : return predicts

    args.py
    βββ args.max_seq_len(default : 20)
    βββ args.n_layers(default : 2)
    βββ args.n_heads(default : 2)
    βββ args.hidden_dim(default : 64)
    ```

<br>
<br>

### LastQuery

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541905-1eaf2080-d065-11eb-995e-3e7fa03907d3.png' height='250px '/>
</div>
<br/>


- Kaggle Riiid AIEd Challenge 2020μ [1st place solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318)μλλ€.
- transformer encoderμ μλ ₯μΌλ‘ sequenceμ λ§μ§λ§ queryλ§ μ¬μ©νμ¬ μκ°λ³΅μ‘λλ₯Ό μ€μ΄κ³ , encoderμ outputμ LSTMμ λ£μ΄ νμ΅νλ λ°©μμ λͺ¨λΈμλλ€.
- **Paper Review :**  [[Last Query Transformer RNN for knowledge tracing λ¦¬λ·°]](https://www.notion.so/Last-Query-Transformer-RNN-for-knowledge-tracing-e0930bfff69b4d2e852de4cbd8e44678)
- **κ΅¬ν**

    ```
    model.py
    βββ class LastQuery
    β   βββ init()
    βββ βββ forward() : return predicts

    args.py
    βββ args.max_seq_len(default : 20)
    βββ args.n_layers(default : 2)
    βββ args.n_heads(default : 2)
    βββ args.hidden_dim(default : 64)
    βββ args.Tfixup(default : False)
    ```

<br>

<br>

### TABNET

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541964-2b337900-d065-11eb-9cad-f8d9c86c5f4e.png' height='250px '/>
</div>
<br/>


- tabular dataμμ MLλͺ¨λΈλ³΄λ€ λ μ°μν μ±λ₯μ λ³΄μ΄λ Deep-learning modelμλλ€.
- dataμμ Sparse instance-wise feature selectionμ μ¬μ©νμ¬ μμ²΄μ μΌλ‘ μ€μν feature μ λ³ν΄λΈ ν νμ΅νλ λ°©μμ μ¬μ©νλ©°, feature μ λ³μ non-linearν processingμ μ¬μ©νμ¬ learning capacityλ₯Ό ν₯μμν΅λλ€.
- Sequentialν multi-step architectureλ₯Ό  κ°μ§κ³ μμΌλ©°, feature maskingμΌλ‘ Unsupervised νμ΅λ κ°λ₯ν©λλ€.
- **Paper Review :** [[Tabnet λΌλ¬Έ λ¦¬λ·°]](https://www.notion.so/Tabnet-298eca48c26a4486a4df8e1586cba2ed)

- **κ΅¬ν**

    ```
    model.py
    βββ class TabNet
    β   βββ TabNetPreTrainer
    β   βββ TabNetClassifier
    β   βββ get_scheduler()
    β   βββ get_optimizer()
    βββ βββ forward() : return models

    trainer.py
    βββ tabnet_run(args, train_data, valid_data)
    βββ get_tabnet_model(args)
    βββ tabnet_inference(args, test_data)

    train.py
    βββ tabnet_run()

    args.py
    βββ args.tabnet_pretrain(default : False)
    βββ args.use_test_to_train(default : False)
    βββ args.tabnet_scheduler(default:'steplr')
    βββ args.tabnet_optimizer(default:'adam')
    βββ args.tabnet_lr(default:2e-2)
    βββ args.tabnet_batchsize(default:16384)
    βββ args.tabnet_n_step(default:5)
    βββ args.tabnet_gamma(default:1.7)
    βββ args.tabnet_mask_type(default:'saprsemax')
    βββ args.tabnet_virtual_batchsize(default:256)
    βββ args.tabnet_pretraining_ratio(default:0.8)
    ```

<br>

## [Input CSV File]

λ°μ΄ν°λ μλμ κ°μ ννμ΄λ©°, ν νμ ν μ¬μ©μκ° ν λ¬Έν­μ νμμ λμ μ λ³΄μ κ·Έ λ¬Έν­μ λ§μ·λμ§μ λν μ λ³΄κ° λ΄κ²¨μ Έ μμ΅λλ€. λ°μ΄ν°λ λͺ¨λ Timestamp κΈ°μ€μΌλ‘ μ λ ¬λμ΄ μμ΅λλ€.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542042-3d151c00-d065-11eb-8278-be19e177037e.png' weight=80%/>
</div>
<br/>


- `userID` μ¬μ©μμ κ³ μ λ²νΈμλλ€. μ΄ 7,442λͺμ κ³ μ  μ¬μ©μκ° μμΌλ©°, train/testμμ μ΄ `userID`λ₯Ό κΈ°μ€μΌλ‘ 90/10μ λΉμ¨λ‘ λλμ΄μ‘μ΅λλ€.
- `assessmentItemID` λ¬Έν­μ κ³ μ λ²νΈμλλ€. μ΄ 9,454κ°μ κ³ μ  λ¬Έν­μ΄ μμ΅λλ€.
- `testId` μνμ§μ κ³ μ λ²νΈμλλ€. λ¬Έν­κ³Ό μνμ§μ κ΄κ³λ μλ κ·Έλ¦Όμ μ°Έκ³ νμ¬ μ΄ν΄νμλ©΄ λ©λλ€. μ΄ 1,537κ°μ κ³ μ ν μνμ§κ° μμ΅λλ€.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542102-49997480-d065-11eb-9957-c84bc1ab77d5.png' height='150px'/>
</div>
<br/>

- `answerCode` μ¬μ©μκ° ν΄λΉ λ¬Έν­μ λ§μ·λμ§ μ¬λΆμ λν μ΄μ§ λ°μ΄ν°μ΄λ©° 0μ μ¬μ©μκ° ν΄λΉ λ¬Έν­μ νλ¦° κ², 1μ μ¬μ©μκ° ν΄λΉ λ¬Έν­μ λ§μΆ κ²μλλ€.
- `Timestamp` μ¬μ©μκ° ν΄λΉλ¬Έν­μ νκΈ° μμν μμ μ λ°μ΄ν°μλλ€.
- `KnowledgeTag` λ¬Έν­ λΉ νλμ© λ°°μ λλ νκ·Έλ‘, μΌμ’μ μ€λΆλ₯ μ­ν μ ν©λλ€. νκ·Έ μμ²΄μ μ λ³΄λ λΉμλ³ν λμ΄μμ§λ§, λ¬Έν­μ κ΅°μ§ννλλ° μ¬μ©ν  μ μμ΅λλ€. 912κ°μ κ³ μ  νκ·Έκ° μ‘΄μ¬ν©λλ€.

## [Feature]

```plain-text
elapsed: μ μ κ° λ¬Έμ λ₯Ό νΈλλ°μ μμν μκ°

time_bin: λ¬Έμ λ₯Ό νΌ μκ°λ(μμΉ¨, μ μ¬, μ λ, μλ²½)

classification: λλΆλ₯(νλ)

paperNum: μνμ§ λ²νΈ

problemNum: λ¬Έμ  λ²νΈ

user_total_acc: μ μ μ μ΄ μ λ΅λ₯ 

test_acc: κ° μνμ§μ νκ·  μ λ΅λ₯ 

assessment_acc: κ° λ¬Έμ μ νκ·  μ λ΅λ₯ 

tag_acc: κ° νκ·Έμ νκ·  μ λ΅λ₯ 

total_used_time: μ μ κ° νλμ μνμ§λ₯Ό λ€ νΈλλ°μ μμν μκ°

past_correct: μ μ λ³ κ³Όκ±° λ§μΆ λ¬Έμ μ μ

past_content_count: μ μ -λ¬Έμ λ³ κ³Όκ±°μ λμΌ λ¬Έμ λ₯Ό λ§λ νμ

correct_per_hour: μκ°(hours)λ³ μ λ΅λ₯ 

same_tag: λμΌ νκ·Έλ₯Ό μ°μμΌλ‘ νμλμ§ μ λ¬΄(T/F)

cont_tag: μ°μμΌλ‘ νΌ λμΌ νκ·Έ κ°μ(0~)

etc...
```

<br>
<br>


## [Contributors]

- **μ ν¬μ** ([Heeseok-Jeong](https://github.com/Heeseok-Jeong))
- **μ΄μ λ** ([Anna-Lee](https://github.com/ceanna93))
- **μ΄μ°½μ°** ([changwoomon](https://github.com/changwoomon))
- **μμ μ§** ([dkswndms4782](https://github.com/dkswndms4782))
- **μ μ¬μ°** ([JAEWOOSUN](https://github.com/JAEWOOSUN))

<br>
<br>

## [Collaborative Works]

**Gitflow λΈλμΉ μ λ΅**

`β 92κ°μ Commits, 26κ°μ Pull Requests`

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542173-59b15400-d065-11eb-9c49-56c091e4fc9f.gif' height='250px '/>
</div>
<br/>
<br>


**Github issues & projects λ‘ μΌμ  κ΄λ¦¬**

`β 28κ°μ Issues`

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542239-6e8de780-d065-11eb-9f6f-821372f4bbbf.gif' height='250px '/>
</div>
<br/>
<br>


`β Modeling Project μμ κ΄λ¦¬`

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542265-751c5f00-d065-11eb-8465-97d2a6fbfebf.gif' height='250px '/>
</div>
<br/>
<br>


**Notion μ€νλΈνΈλ‘ μ€ν κ³΅μ **

`β 39κ°μ μ€νλΈνΈ`

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542297-81082100-d065-11eb-96de-713440c9544b.gif' height='250px '/>
</div>
<br/>
<br>


**Notion μ μΆκΈ°λ‘μΌλ‘ μ μΆ λ΄μ­ κ³΅μ **

`β 155κ°μ μ μΆκΈ°λ‘`

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542323-86fe0200-d065-11eb-8949-8aa146157d73.gif' height='250px '/>
</div>
<br>
<br>


## π Notion

νΌμ΄λ€μ `Ground Rule`, `μ€νλΈνΈ`, `νΌμ΄μΈμ` λ± νλ¬ κ°μ νλ³΄λ₯Ό νμΈνμλ €λ©΄ λ€μ λ§ν¬λ₯Ό ν΄λ¦­νμΈμ.

* **LINK :** [DKT-10μ‘°-No_Caffeine_No_Gain](https://www.notion.so/DKT-10-No_Caffeine_No_Gain-dcc1e3823ec849578ab5ae0bcf117145)


<br>
<br>

## [Reference]

### Papers

- [Deep Knowledge Tracing (Piech et al., arXiv 2015)](https://arxiv.org/pdf/1506.05908.pdf)
- [Last Query Transformer RNN for Knowledge Tracing (Jeon, S., arXiv 2021)](https://arxiv.org/abs/2102.05038)
- [Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing (Choi et al., arXiv 2021)](https://arxiv.org/abs/2002.07033)
- [How to Fine-Tune BERT for Text Classification? (Sun et al., arXiv 2020)](https://arxiv.org/pdf/1905.05583.pdf)
- [Improving Transformer Optimization Through Better Initialization (Huang et al., ICML 2020)](https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf)

### Dataset
- **i-Scream edu Dataset**
<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542423-9ed58600-d065-11eb-9c4e-8c8efa83de80.png' height='120px '/>
</div>
