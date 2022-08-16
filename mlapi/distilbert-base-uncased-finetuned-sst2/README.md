---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
model-index:
- name: distilbert-base-uncased-finetuned-sst2
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: glue
      type: glue
      args: sst2
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.908256880733945
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-finetuned-sst2

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the glue dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4493
- Accuracy: 0.9083

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.1804        | 1.0   | 2105  | 0.2843          | 0.9025   |
| 0.1216        | 2.0   | 4210  | 0.3242          | 0.9025   |
| 0.0871        | 3.0   | 6315  | 0.3320          | 0.9060   |
| 0.0607        | 4.0   | 8420  | 0.3913          | 0.9025   |
| 0.0429        | 5.0   | 10525 | 0.4493          | 0.9083   |


### Framework versions

- Transformers 4.18.0
- Pytorch 1.12.0.dev20220409+cu115
- Datasets 2.0.0
- Tokenizers 0.12.0
