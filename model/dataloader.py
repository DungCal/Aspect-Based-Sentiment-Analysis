import os
import torch
from datasets import load_dataset, DatasetDict

def load_dataset_from_path(save_data_dir, dataset_name, train_file, test_file):
    save_data_dir = os.path.join(save_data_dir, dataset_name)
    print(f"Load data from: {save_data_dir}")

    train_file_path = os.path.join(save_data_dir, train_file)
    train_dataset = load_dataset('csv', data_files=train_file_path)

    # validation_file_path = os.path.join(save_data_dir, validation_file)
    # validation_dataset = load_dataset('json', data_files=validation_file_path)

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