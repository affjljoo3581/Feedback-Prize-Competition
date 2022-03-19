# Feedback Prize - Evaluating Student Writing

## Introduction

This repository contains the code that acheived 4th place in [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/c/feedback-prize-2021/overview). You can see the detailed explanation of 4th place solution [in this post](https://www.kaggle.com/c/feedback-prize-2021/discussion/313330) and also please check [the final private leaderboard](https://www.kaggle.com/c/feedback-prize-2021/leaderboard).

## Requirements
This code requires the below libraries:
* numpy
* omegaconf
* pandas
* pytorch_lightning
* scikit_learn
* sentencepiece
* torch==1.10.2+cu113
* transformers
* iterative-stratification
* text_unidecode
* wandb

Instead of installing the above modules independently, you can simply do at once by using:
```bash
$ pip install -f requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

This repository supports [NVIDIA Apex](https://github.com/NVIDIA/apex). It will automatically detect the apex module and if it is found then some training procedures will be replaced with the highly-optimized and fused operations in the apex module. Run the below codes in the terminal to install apex and enable performance boosting:

```bash
$ git clone https://github.com/NVIDIA/apex
$ sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
$ rm -rf apex
```

## Preparing dataset
Before training the models, you need to download the dataset from [the competition page](https://www.kaggle.com/c/feedback-prize-2021/data). Instead, you can simply download and unzip the dataset by:
```bash
$ pip install --upgrade kaggle

$ export KAGGLE_USERNAME=[your kaggle username]
$ export KAGGLE_KEY=[your kaggle api key]
    
$ kaggle competitions download -c feedback-prize-2021
$ unzip -qq feedback-prize-2021.zip -d feedback-prize-2021
$ rm feedback-prize-2021.zip
```
Make sure the dataset directory name is `feedback-prize-2021`.

## Train your model
In this repository, we provide 4 finetuning configurations which are used in the solution:
* deberta-large.yaml
* deberta-v2-xlarge.yaml
* deberta-v3-large.yaml
* deberta-xlarge.yaml

Of course you can write your own finetuning configuration:
```yaml
dataset:
  dataset_filename: ./feedback-prize-2021/train.csv
  textfile_dir: ./feedback-prize-2021/train
  max_length: 2048
  normalize: true
  num_folds: 5
  fold_index: 0
  dataloader_workers: -1
  random_seed: 42

model:
  transformer:
    pretrained_model_name_or_path: ...
  decoding:
    beam_size: 4
    minimum_lengths:
      Lead: 9
      Claim: 3
      Position: 5
      Rebuttal: 4
      Evidence: 14
      Counterclaim: 6
      Concluding Statement: 11
    minimum_probs:
      Lead: 0.70
      Claim: 0.55
      Position: 0.55
      Rebuttal: 0.55
      Evidence: 0.65
      Counterclaim: 0.50
      Concluding Statement: 0.70
  num_reinit_layers: 0
  random_seed: 42

optim:
  optimizer:
    lr: ...
    betas: [0.9, 0.999]
    eps: 1e-6
    weight_decay: 0.01
  scheduler:
    name: linear
    num_warmup_steps: ...
    num_training_steps: ...

train:
  name: ...
  batch_size: ...
  accumulate_grads: ...
  max_grad_norm: ...
  gradient_checkpointing: ...
  validation_interval: ...
  logging_interval: 10
  evaluate_after_steps: ...
  save_best_checkpoint: true
  precision: 16
  gpus: 1
```
Here are descriptions for main hyperparameters:
* `model.transformer.pretrained_model_name_or_path`: name of the backbone transformer
* `optim.optimizer.lr`: learning rate of the optimizer
* `optim.scheduler.num_warmup_steps`: warmup steps for linear learning rate decay
* `optim.scheduler.num_training_steps`: total training steps
* `train.name`: name of the finetuning experiment
* `train.batch_size`: batch size for single training step
* `train.accumulate_grads`: number of gradient accumulation steps
* `train.max_grad_norm`: maximum gradient norm
* `train.gradient_checkpointing`: boolean whether to determine gradient checkpointing
* `train.validation_interval`: interval of validations per epoch
* `train.evaluate_after_steps`: validations will be performed after this steps

After writing your own configuration, run the below code to train the model:
```bash
$ python src/train.py config/... dataset.num_folds=... dataset.fold_index=...
```
Note that you can change the hyperparameters in command line. It is useful to change the target fold index `dataset.fold_index=...`.

In addition, we recommend to login [wandb](https://wandb.ai/) to log the metrics.

## Inference
If you complete the training and prepare the model weights, now it is time to create a submission.
```bash
$ python src/predict.py ... --output_name=submission.csv --textfile_dir=... --batch_size=4 --max_length=1024 --beam_size=4
```
You have to specify the test directory path to `--textfile_dir=...` option. The prediction script supports simple model ensemble and the predictions will be averaged before extracting the entities. We strongly recommend to predict the model separately and ensemble them using below section. `--return_confidence` option makes the submission to contain the confidence of each entity. It is required when you are using the below ensemble script.

## Entity-level group ensemble
You can ensemble the predictions to improve the performance. Because the tokenizers of different architectures have different subword tokenizations, it is impossible to apply the simple average ensemble. This script gathers the similar entities and sorts by their confidence and average the ranges in each group. If you want to use this script, make sure the submission files contain `confidence` column which is produced by `--return_confidence` option in the prediction script.
```bash
$ python src/ensemble.py \
    deberta-large-fold0.csv \
    deberta-large-fold1.csv \
    deberta-large-fold2.csv \
    deberta-large-fold3.csv \
    deberta-large-fold4.csv \ 
    deberta-v3-large-fold0.csv \
    deberta-v3-large-fold1.csv \
    deberta-v3-large-fold2.csv \
    deberta-v3-large-fold3.csv \
    deberta-v3-large-fold4.csv \
    --min_threshold=5
```
In our experiments, `--group_strategy=mean` with `--min_threshold=[half of submissions]` performed best.
