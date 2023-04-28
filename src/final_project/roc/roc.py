from typing import Dict
from sklearn.metrics import auc, f1_score, precision_recall_curve

def get_f1_auc(y_test, y_pred) -> Dict[str, float]:
    return {
        "f1": f1_score(y_test, y_pred), 
        "auc": auc()
    }
