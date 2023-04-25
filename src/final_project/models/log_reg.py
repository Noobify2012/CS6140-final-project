from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List


def get_lr_pipeline() -> Pipeline:
    return Pipeline(steps=[("lr", LogisticRegression())])


# def get_lr_param_grid() -> List[Dict]:
#     return [
#         {
#             "lr__solver": ["liblinear"],
#             "lr__C": [00.1, 0.1, 0.5, 0.1, 1, 5, 10],
#             "lr__penalty": ["l1", "l2"],
#         },
#         {
#             "lr__solver": ["sage"],
#             "lr_C": [00.1, 0.1, 0.5, 0.1, 1, 5, 10],
#             "lr__penalty": ["l2", "elasticnet"],
#         },
#     ]


def get_lr_param(
    solver: str, 
    penalty: List[str],
    c_list: List[float] = [.001, .01, .5, .1, 1, 5, 10],
) -> Dict[str, List[Any]]:
    return {"lr__solver": list(solver), "lr__C": c_list, "lr__penalty": penalty}


def get_lr_gridsearchcv(
    pipeline: Pipeline,
    param_grid: List[Dict],
    scoring: str = "f1",
    cv: int = 5,
    verbose: int = 0,
    n_jobs: int = 4,
) -> GridSearchCV:
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )
