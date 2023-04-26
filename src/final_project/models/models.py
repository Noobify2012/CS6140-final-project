import joblib

from enum import Enum
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Any, Dict, List

model_dir = Path.cwd().parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)

class ModelENUM(Enum):
    """Simple enum for different model types"""
    SVM = ("SVM")
    LR = ("LR")

    def __init__(self, title: str):
        self._title = title

    @property
    def title(self) -> str:
        return self._title
    


def get_pipeline(model_type: ModelENUM) -> Pipeline:
    """Returns a pipeline appropriate for the given model"""
    if model_type is ModelENUM.SVM:
        steps = [("scaler", StandardScaler()), ("svm", SVC())]
    elif model_type is ModelENUM.LR:
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
    """Return the the parameters to iterate over for a GridSearchCV for SVM"""
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
    """Return the the parameters to iterate over for a GridSearchCV for Logistic Regression"""
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
    """Return a properly formed GridSearchCV to run"""
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
    """Get a dictionary of the best performing  parameters for the model"""
    params = model.best_estimator_.get_params()
    if model_type is ModelENUM.LR:
        return {
            "solver": params["lr__solver"],
            "penalty": params["lr__penalty"],
            "C": params["lr__C"],
        }
    elif model_type is ModelENUM.SVM:
        return {
            "gamma": params["svm__gamma"],
            "kernel": params["svm__kernel"],
            "C": params["svm__C"],
            "penalty": params["svm__penalty"],
        }
    else:
        raise ValueError(
            f"ModelENUM value {model_type} has not been accounted for"
        )

def save_model(model_type: ModelENUM, model: GridSearchCV) -> None:
    """Save the best performing model"""
    best_estimator = model.best_estimator_
    best_params = get_best_params(model_type, model)
    file_string = f"{model_type.title}"
    for k, v in best_params.items():
        if k == "C":
            file_string += "_" + f"C{v}"
        else:
            file_string += "_" + f"{v}"
    file_string += ".pkl"
    joblib.dump(best_estimator, model_dir / file_string)

def analyze_model(model, x_test, x_train, y_test, y_train):
    test_accuracy = model.score(x_test, y_test)
    train_accuracy = model.score(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    prf = precision_recall_fscore_support(y_test, y_pred, average='binary') # TODO different average values: micro macro binary weighted samples

    print(f"Test accuracy: {test_accuracy}")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Precision: {prf[0]}")
    print(f"Recall: {prf[1]}")
    print(f"F-Beta Score: {prf[2]}")
    print(f"F1 Score: {f1}")