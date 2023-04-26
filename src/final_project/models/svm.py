from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List


def get_svm_pipeline() -> Pipeline:
    return Pipeline(steps=[("scaler", StandardScaler()), ("svm", SVC())])


def get_svm_param(
    gamma: List[str] = ['scale', 'auto'],
    c_list: List[float] = [.1, 1, 10, 100],
    coef0: List[float] = [0.0, .01, .1, 1],
    kernel: List[str] = ['poly', 'sigmoid'],
) -> Dict[str, List[ANY]]:
    return {
        'svm__gamma': gamma,
        'svm__C': c_list,
        'svm__coef0': coef0,
        'svm__kernel': kernel,
    }

def get_svm_gridsearchsv(
        pipeline: Pipeline,
        param_grid: List[Dict],
        scoring: str = "f1",
        cv: int = 5,
        verbose: int = 10,
        n_jobs: int = 4,
) -> GridSearchCV:
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs
    )