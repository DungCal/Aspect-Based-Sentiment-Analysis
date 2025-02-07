import os
import json
import logging
from functools import partial
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from dataclasses import field,dataclass
import transformers
import torch
from torch.utils.data import Dataset, DataLoader

import evaluate
import datasets
from datasets import DatasetDict, load_dataset

from transformers import HfArgumentParser,TrainingArguments,AutoConfig,AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,EvalPrediction,Trainer, default_data_collator,set_seed

from sklearn.model_selection import train_test_split
from transformers.trainer_utils import IntervalStrategy, HubStrategy
from huggingface_hub import login,create_repo



os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)



@dataclass
class DataTrainingArguments:
  dataset_name: Optional[str] = field(
      default="", metadata={"help": "The name of the dataset to use (via the datasets library)."}
  )
  save_data_dir: Optional[str] = field(
      default="data", metadata={"help": "A folder save the dataset."}
  )
  train_file: Optional[str] = field(
      default="restaurants_train.csv", metadata={"help": "A csv or a json file containing the training data."}
  )
  # validation_file: Optional[str] = field(
  #     default="validation.jsonl", metadata={"help": "A csv or a json file containing the validation data."}
  # )
  test_file: Optional[str] = field(
      default="restaurants_test.csv", metadata={"help": "A csv or a json file containing the test data."}
  )
  text_column_name: Optional[str] = field(
      default="review_body",
      metadata={
          "help": (
              "The name of the text column in the input dataset or a CSV/JSON file."
              'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
          )
      },
  )
  text_column_delimiter: Optional[str] = field(
      default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
  )
  label_column_name: Optional[str] = field(
      default="feeling",
      metadata={
          "help": (
              "The name of the label column in the input dataset or a CSV/JSON file."
              'If not specified, will use the "label" column for single/multi-label classifcation task'
          )
      },
  )
  max_seq_length: int = field(
      default=128,
      metadata={
          "help": (
              "The maximum total input sequence length after tokenization. Sequences longer "
              "than this will be truncated, sequences shorter will be padded."
          )
      },
  )
  overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
  )
  pad_to_max_length: bool = field(
      default=True,
      metadata={
          "help": (
              "Whether to pad all samples to `max_seq_length`. "
              "If False, will pad the samples dynamically when batching to the maximum length in the batch."
          )
      },
  )
  shuffle_train_dataset: bool = field(
      default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
  )
  shuffle_seed: int = field(
      default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
  )
  max_train_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
          )
      },
  )
  max_eval_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
          )
      },
  )
  metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})



@dataclass
class ModelArguments:
  model_name_or_path: str = field(
      default="bert-base-uncased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None,
      metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
  )
  use_fast_tokenizer: bool = field(
      default=True,
      metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
  )



@dataclass
class TrainingArgumentsCustom(TrainingArguments):
  output_dir: str = field(
      default="outputs",
      metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
  )
  per_device_train_batch_size: int = field(
      default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
  )
  per_device_eval_batch_size: int = field(
      default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
  )
  do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
  do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
  do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})
  num_train_epochs: float = field(
      default=5.0, metadata={"help": "Total number of training epochs to perform."}
  )
  logging_dir: Optional[str] = field(
      default="logs", metadata={"help": "Tensorboard log dir."}
  )
  logging_strategy: Union[IntervalStrategy, str] = field(
      default="steps",
      metadata={"help": "The logging strategy to use."},
  )
  logging_steps: float = field(
      default=500,
      metadata={
          "help": (
              "Log every X updates steps. Should be an integer or a float in range `[0,1)`."
              "If smaller than 1, will be interpreted as ratio of total training steps."
          )
      },
  )
  evaluation_strategy: Union[IntervalStrategy, str] = field(
      default="epoch",
      metadata={"help": "The evaluation strategy to use."},
  )
  save_strategy: Union[IntervalStrategy, str] = field(
      default="epoch", metadata={"help": "The checkpoint save strategy to use."},
  )
  save_steps: float = field(
      default=100,
      metadata={
          "help": (
              "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
              "If smaller than 1, will be interpreted as ratio of total training steps."
          )
      },
  )
  save_total_limit: Optional[int] = field(
      default=2, metadata={"help": ("If a value is passed, will limit the total amount of checkpoints.")},
  )
  load_best_model_at_end: Optional[bool] = field(
      default=True,
      metadata={
          "help": (
              "Whether or not to load the best model found during training at the end of training. When this option"
              " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
          )
      },
  )
  metric_for_best_model: Optional[str] = field(
      default="accuracy", metadata={"help": "The metric to use to compare two different models."}
  )
  report_to: Optional[List[str]] = field(
      default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."}
  )
  optim: str = field(
      default="adamw_torch",
      metadata={"help": "The optimizer to use."},
  )
  bf16: bool = field(
      default=False,
      metadata={
          "help": (
              "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
              " architecture or using CPU (use_cpu). This is an experimental API and it may change."
          )
      },
  )
  fp16: bool = field(
      default=False,
      metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
  )
  push_to_hub: bool = field(
      default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
  )
  hub_strategy: Union[HubStrategy, str] = field(
      default="every_save",
      metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
  )
  hub_model_id: Optional[str] = field(
      default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
  )
  hub_token: Optional[str] = field(
      default=None, metadata={"help": "The token to use to push to the Model Hub."}
  )



# def load_dataset_from_datahub(dataset_name,save_data_dir):
#   raw_dataset=load_dataset(dataset_name)
#   for dataset_type in raw_dataset:
#     examples=raw_dataset[dataset_type]
#     sentences=[]
#     sentences=examples['text']
#     labels=examples['feeling']
#     labels=[id2label[str(idx)] for idx in labels]

#     save_data_file=os.path.join(save_data_dir,f'{dataset_type}.jsonl')
#     print(f'Write into...{save_data_file}')
#     with open(save_data_file,'w') as f:
#       for sentence,label in zip(sentences,labels):
#         data={'sentence':sentence,'label':label}
#         print(json.dumps(data,ensure_ascii=False),file=f)


def load_dataset_from_path(save_data_dir, dataset_name, train_file, test_file):
    save_data_dir = os.path.join(save_data_dir, dataset_name)
    print(f"Load data from: {save_data_dir}")

    train_file_path = os.path.join(save_data_dir, train_file)
    train_dataset = load_dataset('csv', data_files=train_file_path)

    test_file_path = os.path.join(save_data_dir, test_file)
    test_dataset = load_dataset('csv', data_files=test_file_path)

    train_test_split = test_dataset['train'].train_test_split(test_size=0.5, shuffle=True, seed=42)

    # Tạo DatasetDict mới với eval và test
    eval_test_dataset = DatasetDict({
        'eval': train_test_split['train'],
        'test': train_test_split['test']
    })

    data = {
        'train': train_dataset['train'],
        'validation': eval_test_dataset['eval'],
        'test': eval_test_dataset['test']
    }

    return data


def preprocess_function(example, tokenizer):

    tokens=example['Tokens'].replace("'",'').strip('][').split(', ')
    tags=example['Tags'].strip('][').split(', ')
    pols=example['Polarities'].strip('][').split(', ')

    bert_tokens=[]
    pol_tensors=[]
    bert_att=[]
    pols_label=0

    for i in range(len(tokens)):
        #tao token cho tung word(subword) roi append vao list
        t=tokenizer.tokenize(tokens[i])
        bert_tokens+=t
        if int(pols[i])!=-1:
            bert_att+=t
            pols_label=int(pols[i])

    #convert token to ids theo tu dien
    segment_tensor=[0]+[0]*len(bert_tokens)+[0]+[1]*len(bert_att)
    bert_tokens=['[CLS]']+bert_tokens+['[SEP]']+bert_att
    bert_tokens=' '.join(bert_tokens)
    #print(bert_tokens)
    result=tokenizer(bert_tokens,padding='max_length',max_length=128,truncation=True,add_special_tokens=False)


    #ids_tensor=torch.tensor(bert_ids)
    #segment_tensor=torch.tensor(segment_tensor)
    pols_tensor=torch.tensor(pols_label)
    result['bert_tokens']=bert_tokens
    result['segment_tensors']=segment_tensor + [0]*(128-len(segment_tensor))
    result['labels']=pols_tensor

    return result


metric=evaluate.load('accuracy')

def compute_metrics(eval_pred:EvalPrediction):
  predictions,labels=eval_pred
  predictions=np.argmax(predictions,1)
  result=metric.compute(predictions=predictions,references=labels)

  return result


def do_nothing(example):
  return example



import sys

def train(model_args, data_args, training_args):

  # Setup logging
  # thiet lap logging co ban: dinh dang va cau hinh
  logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #thiet lap muc do logging
  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
    # thiet lap logging cho cac thu vien  con
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    #thiet lap cac thong tin huan luyen vao log nhu device, gpu,
    #những thông tin này có thể quan trọng và cần được chú ý
  logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
  #ghi lại các tham số huấn luyện/đánh giá (training_args) với mức độ "INFO
  logger.info(f"Training/evaluation parameters {training_args}")


  set_seed(training_args.seed)
    # login hub
  if training_args.push_to_hub:
    login(
            token=training_args.hub_token
        )

  try:
    create_repo(training_args.hub_model_id, private=False)
  except:
    pass

  #load dataset
  raw_dataset=load_dataset_from_path(data_args.save_data_dir,data_args.dataset_name,data_args.train_file,data_args.test_file)
  raw_dataset=DatasetDict(raw_dataset)
  num_labels=3
  label2id={'negative':0,'neutral':1,'positive':2}
  id2label={0:'negative',1:'neutral',2:'positive'}


  #ghi lai thong tin raw dataset
  logger.info(f'Dataset loaded: {raw_dataset}')

  #load pretrained model and tokenizer
  tokenizer=AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                          #num_labels=num_labels,
                                          #label2id=label2id,
                                          #id2label=id2label,
                                          cache_dir=model_args.cache_dir,
                                          use_fast=model_args.use_fast_tokenizer)


  config=AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                    num_labels=3,
                                    finetuning_task='text-classification',
                                    cache_dir=model_args.cache_dir)

  model=AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                           from_tf=bool('.ckpt' in model_args.model_name_or_path),
                                                           #num_labels=num_labels,
                                                           config=config,
                                                           cache_dir=model_args.cache_dir)
  #để đảm bảo rằng một số tác vụ nhất định (như tải hoặc xử lý dữ liệu) chỉ được thực hiện bởi một tiến trình duy nhất trong trường hợp bạn đang sử dụng huấn luyện phân tán
  #desc: Tham số desc trong ví dụ này được đặt là "Running tokenizer on dataset". Khi tiến trình chính thực hiện công việc tiền xử lý dữ liệu, mô tả này sẽ xuất hiện trong
  #các log hoặc các thông báo, giúp bạn biết rằng tiến trình chính đang thực hiện việc tokenization trên tập dữ liệu.
  # with training_args.main_process_first(desc='Dataset map preprosessing'):
  processed_dataset=raw_dataset.map(partial(preprocess_function,tokenizer=tokenizer), #create a new function where data_args, tokenizer, and label2id are already provided
                                      load_from_cache_file=not data_args.overwrite_cache,
                                      desc='Running tokenize on dataset')

  processed_dataset=processed_dataset.map(do_nothing,batched=True)




  if data_args.pad_to_max_length:
    data_collator=default_data_collator
  else:
    data_collator=None

  #Trainer
  print('cc')
  trainer=Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=processed_dataset['train'],
                  eval_dataset=processed_dataset['validation'],
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)
  print('training...')
  if training_args.do_train:
    #return processed_dataset
    train_result=trainer.train()
    metrics=train_result.metrics
    metrics['train_samples']=len(processed_dataset['train'])
    trainer.log_metrics('train',metrics)
    trainer.save_metrics('train',metrics)
    trainer.save_state()
  print('eval...')
  if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics=trainer.evaluate(eval_dataset=processed_dataset['validation'])
    metrics['eval_samples']=len(processed_dataset['validation'])
    trainer.log_metrics('eval',metrics)
    trainer.save_metrics('eval',metrics)
  print('predict...')
  if training_args.do_predict and processed_dataset['test'] is not None:
    logger.info("*** Predict ***")
    if 'labels' in processed_dataset['test'].features:
      metrics=trainer.evaluate(eval_dataset=processed_dataset['test'])
      metrics['test_samples']=len(processed_dataset['test'])
      trainer.log_metrics('eval',metrics)
      trainer.save_metrics('eval',metrics)

    predictions=trainer.predict(test_dataset=processed_dataset['test'],metric_key_prefix='predict').predictions
    predictions=np.argmax(predictions,axis=1)
    output_predict_file=os.path.join(training_args.output_dir,'predict_results.txt')
    with open(output_predict_file,'w') as writer:
      logger.info("***** Predict results *****")
      writer.write('index\tprediction\n')
      for index,item in enumerate(predictions):
        item=id2label[item]
        writer.write(f'{index}\t{item}\n')

    logger.info(f'Predicted results saved at: {output_predict_file}')

    print('het vi')
    tokenizer.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
      trainer.create_model_card()
      trainer.push_to_hub()



def main():
  parser=HfArgumentParser((ModelArguments,DataTrainingArguments,TrainingArgumentsCustom))

  def filter_args(args):
    # Chỉ giữ lại các đối số có tiền tố là '--'
    return [arg for arg in args if arg.startswith("--")]

# Nếu không có đối số dòng lệnh, sử dụng giá trị mặc định
  if len(sys.argv) == 1:
      model_args, data_args,training_args = parser.parse_args_into_dataclasses(args=[])
  else:
      filtered_args = filter_args(sys.argv)
      model_args, data_args ,training_args = parser.parse_args_into_dataclasses(filtered_args)

  #return model_args,data_args,training_args

  os.makedirs(training_args.output_dir,exist_ok=True)

  save_model_path=model_args.model_name_or_path.split('/')[-1]+'-'+data_args.dataset_name.replace('_','-')
  training_args.hub_model_id=model_args.model_name_or_path.split('/')[-1]+'-'+data_args.dataset_name.replace('_','-')
  if training_args.fp16:
    training_args.model_hub_id=training_args.model_hub_id+'-fp16'
    save_model_path=save_model_path+'-fp16'
  elif training_args.bf16:
    training_args.model_hub_id=training_args.model_hub_id+'-bf16'
    save_model_path=save_model_path+'-bf16'

  training_args.output_dir=os.path.join(training_args.output_dir,save_model_path)
  if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir)
  training_args.logging_dir=os.path.join(training_args.output_dir,training_args.output_dir)

  train(model_args,data_args,training_args)

if __name__=="__main__":
  main()
