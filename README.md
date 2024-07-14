# Aspect-Based-Sentiment-Analysis
Aspect-Based Sentiment Analysis using Fine-Tuning BERTs 

## Dependencies
- Python 3.10
 [Pytorch](https://github.com/pytorch/pytorch) 2.0+
```
pip install -r requirements.txt
```

## Dataset
[Sentiment-analysis](https://drive.google.com/uc?id=1d7JABk4jViI-USjLsWmhGkvzi8uQIL5C)

## Fine-tuning BERTs
```
    python run_fine_tuning_bert.py \
        --dataset_name semeval-task-4-2014 \
        --model_name_or_path bert-base-uncased \
        --do_train True \
        --do_eval True \
        --do_predict True
```

### Predict
