from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List


class ModelENUM(Enum):
    SVM = 1
    LR = 2


def get_pipeline(model_type: ModelENUM) -> Pipeline:
    if model_type == ModelENUM.SVM:
        steps = [("scaler", StandardScaler()), ("svm", SVC())]
    elif model_type == ModelENUM.LR:
        steps = [("lr", LogisticRegression())]
    else:
        raise ValueError(
            f"ModelENUM value {model_type} has not been accounted for"
        )
    return Pipeline(steps=steps)


def get_svm_param(
    gamma: List[str] = ["scale", "auto"],
    c_list: List[float] = [0.001, 0.01, 0.1, 1, 10],
    coef0: List[float] = [0.0, 0.01, 0.1, 1],
    kernel: List[str] = ["poly", "sigmoid"],
) -> Dict[str, List[Any]]:
    return {
        "svm__gamma": gamma,
        "svm__C": c_list,
        "svm__coef0": coef0,
        "svm__kernel": kernel,
    }


def get_lr_param(
    solver: str = "liblinear",
    penalty: List[str] = ["l2"],
    c_list: List[float] = [0.001, 0.01, 0.1, 1, 10],
    max_iter: List[int] = [200],
) -> Dict[str, List[Any]]:
    return {
        "lr__solver": [solver],
        "lr__C": c_list,
        "lr__penalty": penalty,
        "lr__max_iter": max_iter,
    }


def get_grid_search_cv(
    pipeline: Pipeline,
    param_grid: List[Dict],
    scoring: str = "f1",
    cv: int = 3,
    verbose: int = 10,
    n_jobs: int = 20,
) -> GridSearchCV:
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )


def get_best_params(
    model_type: ModelENUM, model: GridSearchCV
) -> Dict[str, Any]:
    best_estimator = model.best_estimator_.get_params()
    if model_type == ModelENUM.LR:
        return {
            "solver": best_estimator["lr__solver"],
            "penalty": best_estimator["lr__penalty"],
            "C": best_estimator["lr__C"],
        }
    elif model_type == ModelENUM.SVM:
        return {
            "gamma": best_estimator["svm__gamma"],
            "kernel": best_estimator["svm__kernel"],
            "C": best_estimator["svm__C"],
            "penalty": best_estimator["svm__penalty"],
        }
    else:
        raise ValueError(
            f"ModelENUM value {model_type} has not been accounted for"
        )
