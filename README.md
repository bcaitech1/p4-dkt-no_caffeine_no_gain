# 📝 No Caffeine No Gain
<br>

**프로젝트 기간 : 2021.05.31 ~ 2021.06.15**
<br>
<br>
**프로젝트 내용 : Deep Knowledge Tracing**

<br>

## [목차]

- [\[Deep Knowledge Tracing 소개\]](#deep-knowledge-tracing-소개)
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
  * [📝 Notion](#-notion)
- [[Reference]](#reference)
  * [Papers](#papers)
  * [Dataset](#dataset-1)

<br>
<br>

## [Deep Knowledge Tracing 소개]

**DKT**는 **Deep Knowledge Tracing**의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d7568542-7435-4668-8267-495eaeb5d6ba/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d7568542-7435-4668-8267-495eaeb5d6ba/Untitled.png)

대회에서는 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중합니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/447fe89e-5e3d-4024-ac80-7a125870a8f0/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/447fe89e-5e3d-4024-ac80-7a125870a8f0/Untitled.png)

<br>
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

```bash
pip install -r requirements.txt
```

<br>
<br>

## [Usage]

### Dataset

학습에 필요한 데이터를 만들기 위해 두 개의 `.py` 파일을 순차적으로 실행해야 합니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python make_elapsed.py
$ p4-dkt-no_caffeine_no_gain# python make_fixed_data.py
```

### Train

모델을 학습하기 위해서는 `train.py` 를 실행시킵니다.

아래 Arguments 에 있는 argument 중 필요한 argumet 를 바꿔 사용하면 됩니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python train.py
```

총 7가지의 모델을 선택할 수 있습니다.

- **TABNET**
- **LASTQUERY**
- **SAINT**
- **LGBM**
- **BERT**
- **LSTMATTN**
- **LSTM**

### Inference

학습된 모델로 추론하기 위해서는 `inference.py` 를 실행시킵니다.

필요한 argument 는 `—-model_name` 과 `—-model_epoch` 입니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python inference.py --model_name "학습한 모델 폴더 이름" --model_epoch "사용하고픈 모델의 epoch"
```

### Arguments

train 과 inference 에서 필요한 argument 입니다.

```python
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

## [File Structure]

전체적인 File Structure 입니다.

```
code
├── README.md
├── .gitignore
├── args.py
├── make_custom_data
│   ├── make_elapsed.py - time 관련 feature 생성
│   ├── make_fixed_data.py - user 정답률 기반으로 valid 생성
│   └── make_original_fixed_data.py - shuffle해서 valid 생성
│
├── dkt
│   ├── criterion.py
│   ├── dataloader.py
│   ├── metric.py
│   ├── model.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   └── utils.py
├── ensemble.py
├── inference.py
├── requirements.txt - dependencies
└── train.py

```

<br>
<br>

### LSTM

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2fd4ada4-6d16-4bb5-bf41-2e34795347b4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2fd4ada4-6d16-4bb5-bf41-2e34795347b4/Untitled.png)

- sequence data를 다루기 위한 LSTM 모델입니다.
- **구현**

    ```
    model.py
    ├── class LSTM
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### LSTMATTN

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c335562-4ff2-46b4-a3a5-728021a548e7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c335562-4ff2-46b4-a3a5-728021a548e7/Untitled.png)

- LSTM 모델에 Self-Attention을 추가한 모델입니다.
- **구현**

    ```
    model.py
    ├── class LSTMATTN
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### BERT

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eace3214-70f5-4bc3-9267-f5940d59551c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eace3214-70f5-4bc3-9267-f5940d59551c/Untitled.png)

- `Huggingface` 에서 BERT 구조를 가져와서 사용합니다. 다만, pre-trained 모델이 아니기 때문에 Transformer-encoder 와 같습니다.
- 현재 모델에서는 bert_config 의 is_decoder 를 True 로 주어 Transformer-decoder 로 사용하고 있습니다.
- **구현**

    ```
    model.py
    ├── class Bert
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    ├── args.is_decoder(default : True)
    └── args.hidden_dim(default : 64)
    ****
    ```

<br>
<br>

### LGBM

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cbf5c2b5-1aff-4428-983b-413da3a5ebbe/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cbf5c2b5-1aff-4428-983b-413da3a5ebbe/Untitled.png)

- tabular data에서 좋은 성능을 보이는 Machine Learning 모델입니다.

<br>
<br>

### SAINT

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0682e128-c43a-4940-8481-ffc3faa43e71/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0682e128-c43a-4940-8481-ffc3faa43e71/Untitled.png)

- Kaggle Riiid AIEd Challenge 2020의 [Host가 제시한 solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/193250) 입니다.
- Transformer와 비슷한 구조의 모델로 Encoder와 Decoder를 가지고 있습니다.
- 인코더는 feature 임베딩 스트림에 self-attention 레이어를 적용하고 디코더에서 self-attention 레이어와 인코더-디코더 attention 레이어를 응답 임베딩과 인코더의 출력 스트림에 번갈아 적용하는 구조입니다.
- **Paper Review : [[**Saint 모델 분석]](https://www.notion.so/Saint-507d13692825492ba05128f4548c2da7)
- **구현**

    ```
    model.py
    ├── class Saint
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### LastQuery

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d4beb986-59f0-4b68-8156-5dd6ee283256/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d4beb986-59f0-4b68-8156-5dd6ee283256/Untitled.png)

- Kaggle Riiid AIEd Challenge 2020의 [1st place solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318)입니다.
- transformer encoder의 입력으로 sequence의 마지막 query만 사용하여 시간복잡도를 줄이고, encoder의 output을 LSTM에 넣어 학습하는 방식의 모델입니다.
- **Paper Review :**  [[Last Query Transformer RNN for knowledge tracing 리뷰]](https://www.notion.so/Last-Query-Transformer-RNN-for-knowledge-tracing-e0930bfff69b4d2e852de4cbd8e44678)
- **구현**

    ```
    model.py
    ├── class LastQuery
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    ├── args.hidden_dim(default : 64)
    └── args.Tfixup(default : False)
    ```

<br>

<br>

### TABNET

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a4d49173-70b9-43b1-91f3-3051250e5e4d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a4d49173-70b9-43b1-91f3-3051250e5e4d/Untitled.png)

- tabular data에서 ML모델보다 더 우수한 성능을 보이는 Deep-learning model입니다.
- data에서 Sparse instance-wise feature selection을 사용하여 자체적으로 중요한 feature 선별해낸 후 학습하는 방식을 사용하며, feature 선별시 non-linear한 processing을 사용하여 learning capacity를 향상시킵니다.
- Sequential한 multi-step architecture를  가지고있으며, feature masking으로 Unsupervised 학습도 가능합니다.
- **Paper Review : [[Tabnet 논문 리뷰]](https://www.notion.so/Tabnet-298eca48c26a4486a4df8e1586cba2ed)**

- **구현**

    ```
    model.py
    ├── class TabNet
    │   ├── TabNetPreTrainer
    │   ├── TabNetClassifier
    │   ├── get_scheduler()
    │   ├── get_optimizer()
    └── └── forward() : return models

    trainer.py
    ├── tabnet_run(args, train_data, valid_data)
    ├── get_tabnet_model(args)
    └── tabnet_inference(args, test_data)

    train.py
    └── tabnet_run()

    args.py
    ├── args.tabnet_pretrain(default : False)
    ├── args.use_test_to_train(default : False)
    ****├── args.tabnet_scheduler(default:'steplr')
    ****├── args.tabnet_optimizer(default:'adam')
    ****├── args.tabnet_lr(default:2e-2)
    ****├── args.tabnet_batchsize(default:16384)
    ****├── args.tabnet_n_step(default:5)
    ****├── args.tabnet_gamma(default:1.7)
    ├── args.tabnet_mask_type(default:'saprsemax')
    ├── args.tabnet_virtual_batchsize(default:256)
    └── args.tabnet_pretraining_ratio(default:0.8)
    ****
    ```

<br>

## [Input CSV File]

데이터는 아래와 같은 형태이며, 한 행은 한 사용자가 한 문항을 풀었을 때의 정보와 그 문항을 맞췄는지에 대한 정보가 담겨져 있습니다. 데이터는 모두 Timestamp 기준으로 정렬되어 있습니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b52081c-e41b-41eb-9417-dc319de4e93b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b52081c-e41b-41eb-9417-dc319de4e93b/Untitled.png)

- `userID` 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 `userID`를 기준으로 90/10의 비율로 나누어졌습니다.
- `assessmentItemID` 문항의 고유번호입니다. 총 9,454개의 고유 문항이 있습니다.
- `testId` 시험지의 고유번호입니다. 문항과 시험지의 관계는 아래 그림을 참고하여 이해하시면 됩니다. 총 1,537개의 고유한 시험지가 있습니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dc01d187-575e-4c0e-bddb-5eec928e86db/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dc01d187-575e-4c0e-bddb-5eec928e86db/Untitled.png)

- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터이며 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.
- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.
- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 태그 자체의 정보는 비식별화 되어있지만, 문항을 군집화하는데 사용할 수 있습니다. 912개의 고유 태그가 존재합니다.

## [Feature]

```python
elapsed: 유저가 문제를 푸는데에 소요한 시간

time_bin: 문제를 푼 시간대(아침, 점심, 저녁, 새벽)

classification: 대분류(학년)

paperNum: 시험지 번호

problemNum: 문제 번호

user_total_acc: 유저의 총 정답률

test_acc: 각 시험지의 평균 정답률

assessment_acc: 각 문제의 평균 정답률

tag_acc: 각 태그의 평균 정답률

total_used_time: 유저가 하나의 시험지를 다 푸는데에 소요한 시간

past_correct: 유저별 과거 맞춘 문제의 수

past_content_count: 유저-문제별 과거에 동일 문제를 만난 횟수

correct_per_hour: 시간(hours)별 정답률

same_tag: 동일 태그를 연속으로 풀었는지 유무(T/F)

cont_tag: 연속으로 푼 동일 태그 개수(0~)

etc...
```

<br>
<br>


## [Contributors]

- **정희석** ([Heeseok-Jeong](https://github.com/Heeseok-Jeong))
- **이애나** ([Anna-Lee](https://github.com/ceanna93))
- **이창우** ([changwoomon](https://github.com/changwoomon))
- **안유진** ([dkswndms4782](https://github.com/dkswndms4782))
- **선재우** ([JAEWOOSUN](https://github.com/JAEWOOSUN))

<br>
<br>

## [Collaborative Works]

**Gitflow 브랜치 전략**

`→ 92개의 Commits, 26개의 Pull Requests`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51d09511-77c0-4efe-a65b-c706cae75ecd/pr.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51d09511-77c0-4efe-a65b-c706cae75ecd/pr.gif)

<br>

**Github issues & projects 로 일정 관리**

`→ 28개의 Issues`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38005daf-d372-403f-b7b7-5d162d11bc57/ezgif.com-gif-maker.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38005daf-d372-403f-b7b7-5d162d11bc57/ezgif.com-gif-maker.gif)

`→ Modeling Project 에서 관리`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f937a7c8-19d5-4e52-a49a-68d3843be5b6/project.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f937a7c8-19d5-4e52-a49a-68d3843be5b6/project.gif)

<br>

**Notion 실험노트로 실험 공유**

`→ 39개의 실험노트`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0280f6aa-f632-4155-b7b8-c3d5b7b02a8b/.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0280f6aa-f632-4155-b7b8-c3d5b7b02a8b/.gif)

<br>

**Notion 제출기록으로 제출 내역 공유**

`→ 155개의 제출기록`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc10b2d9-5768-4fab-98ac-a76a131fd492/.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc10b2d9-5768-4fab-98ac-a76a131fd492/.gif)

<br>

### 📝 Notion

피어들의 `Ground Rule`, `실험노트`, `피어세션` 등 한달 간의 행보를 확인하시려면 다음 링크를 클릭하세요.

* LINK : [DKT-10조-No_Caffeine_No_Gain](https://www.notion.so/DKT-10-No_Caffeine_No_Gain-dcc1e3823ec849578ab5ae0bcf117145)

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
- i-Scream edu Dataset
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59ea867b-83f2-4bba-ae0d-e40cadd59c18/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59ea867b-83f2-4bba-ae0d-e40cadd59c18/Untitled.png)
