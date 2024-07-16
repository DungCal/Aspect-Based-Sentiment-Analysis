import os
import sys
import logging
from functools import partial
import numpy as np

import datasets
from datasets import DatasetDict

import transformers
from transformers import (
    AutoTokenizer,
    default_data_collator,
    Trainer,
    set_seed,
)

print("Hello world")

from huggingface_hub import login, create_repo, delete_repo
from model.metric import compute_metrics
from model.dataloader import load_dataset_from_path, preprocess_function
from model.model import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logger = logging.getLogger(__name__)

    

def train(model_args, data_args, training_args):

  # Setup logging
  # thiet lap logging co ban: dinh dang va cau hinh
  logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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


  raw_dataset=load_dataset_from_path(data_args.save_data_dir,data_args.dataset_name,data_args.train_file,data_args.test_file)
  raw_dataset=DatasetDict(raw_dataset)
  num_labels=3
  label2id={'negative':0,'neutral':1,'positive':2}
  id2label={0:'negative',1:'neutral',2:'positive'}


  #ghi lai thong tin raw dataset
  logger.info(f'Dataset loaded: {raw_dataset}')


  #load pretrained model and tokenizer
  tokenizer=AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                          cache_dir=model_args.cache_dir,
                                          use_fast=model_args.use_fast_tokenizer)
  
  model = load_model(model_args)

  with training_args.main_process_first(desc='Dataset map preprosessing'):
    processed_dataset=raw_dataset.map(partial(preprocess_function,data_args=data_args,tokenizer=tokenizer,label2id=label2id), #create a new function where data_args, tokenizer, and label2id are already provided
                                      batched=True,
                                      load_from_cache_file=not data_args.overwrite_cache,
                                      desc='Running tokenize on dataset')


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
    if 'label' in processed_dataset['test'].features:
      metrics=trainer.evaluate(eval_dataset=processed_dataset['test'])
      metrics['test_samples']=len(processed_dataset['test'])
      trainer.log_metrics('eval',metrics)
      trainer.save_metrics('eval',metrics)

    predictions=trainer.predict(test_dataset=processed_dataset['test'],metric_key_prefix='predict').predictions
    predictions=np.argmax(predictions,axis=1)
    output_predict_file=os.path.join(training_args.ouput_dir,'predict_results.txt')
    with open(output_predict_file,'w') as writer:
      logger.info("***** Predict results *****")
      writer.write('index\tprediction\n')
      for index,item in enumerate(predictions):
        item=id2label[item]
        writer.write(f'{index}\t{item}\n')

    logger.info(f'Predicted results saved at: {output_predict_file}')

    print('het vi')
    tokenizer.save_pretrained(training_args.ouput_dir)
    if training_args.push_to_hub:
      trainer.create_model_card()
      trainer.push_to_hub()