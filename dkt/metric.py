from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return auc, acc