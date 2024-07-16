import numpy as np
import evaluate
from transformers import EvalPrediction


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = metric.compute(predictions=predictions, references=labels)
    return result