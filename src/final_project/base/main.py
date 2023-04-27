"""Train models.

Usage:
    train.py lr
    train.py svm
    train.py ffn

Options:
    -h --help     Show this screen.
"""

import matplotlib as plt
import pandas as pd

from docopt import docopt
from final_project import loader, builder, models, plots
from final_project.models import ModelENUM
from pathlib import Path
from torch.nn import Module

# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

# from sklearn.svm import SVC
from typing import List, Tuple, Type

_model_dir = Path.cwd() / "models" / "trained_models"
_model_dir.mkdir(parents=True, exist_ok=True)


def get_massaged_df() -> pd.DataFrame:
    df = loader.get_df(all_files=True)
    df = df[df["Origin"] == "BOS"]
    df = df[
        [
            "DistanceGroup",
            "DayofMonth",
            "Month",
            "Year",
            "Duplicate",
            "ArrDel15",
            "DistanceGroup",
            "WeatherDelay",
            "NASDelay",
            "SecurityDelay",
            "Operating_Airline",
            "Dest",
        ]
    ]
    df = builder.encodeFrame(df)
    return df


def get_test_train(df: pd.DataFrame) -> Tuple[List, List, List, List]:
    x = df.drop(columns=["ArrDel15"])
    y = df[["ArrDel15"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=150
    )
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    return x_train, x_test, y_train, y_test


def train_lr(x_train: List, y_train: List) -> Type[GridSearchCV]:
    lr_pipeline = models.get_pipeline(ModelENUM.LR)
    param_grid = [
        models.get_lr_param(
            solver="liblinear", penalty=["l2"], c_list=[1], max_iter=[5000]
        )
    ]
    grid_model = models.get_grid_search_cv(
        pipeline=lr_pipeline, param_grid=param_grid, cv=5, n_jobs=30
    )
    grid_model.fit(x_train, y_train)
    models.save_model(ModelENUM.LR, grid_model)
    return grid_model


def train_svm(x_train: List, y_train: List) -> Type[GridSearchCV]:
    svc_pipeline = models.get_pipeline(ModelENUM.SVM)
    param_grid = [
        models.get_svm_param(
            gamma=["auto"],
            c_list=[10],
            kernel=["linear"],
            max_iter=[5000],
            cache_size=[8000],
        )
    ]
    grid_model = models.get_grid_search_cv(
        pipeline=svc_pipeline, param_grid=param_grid, cv=5, n_jobs=30
    )
    grid_model.fit(x_train, y_train)
    models.save_model(ModelENUM.SVM, grid_model)
    return grid_model


def train_ffn(x_train: List, y_train: List) -> Type[Module]:
    print("Training feedforward network model...")


def save_plots(
    model_type: ModelENUM,
    file_name: str,
    x_test: List,
    y_test: List,
    model: Type[GridSearchCV],
) -> None:
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)
    plots.save_precision_recall_curve(
        model_type.title, f"{file_name}_prc", y_test, y_prob
    )
    plots.save_confusion_matrix(model_type.title, f"{file_name}_cm", y_test, y_pred)


def run() -> None:
    arguments = docopt(__doc__)

    x_train, x_test, y_train, y_test = get_test_train(get_massaged_df())
    if arguments["lr"]:
        model = train_lr(x_train, y_train)
        save_plots(ModelENUM.LR, "lr_best_model", x_test, y_test, model)
    elif arguments["svm"]:
        model = train_svm(x_train, y_train)
        save_plots(ModelENUM.SVM, "svm_best_model", x_test, y_test, model)
    elif arguments["ffn"]:
        model = train_ffn(x_train, y_train)
