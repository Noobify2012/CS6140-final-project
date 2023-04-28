"""Train models.

Usage:
    final-project lr     
    final-project svm    
    final-project ffn    

Options:
    -h --help     Show this screen.
"""

import joblib
import json
import pandas as pd
import time

from docopt import docopt
from final_project import loader, builder, models, plots
from final_project.models import ModelENUM

from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC
from torch.nn import Module
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
    lr = LogisticRegression(
        solver="saga",
        penalty="l1",
        fit_intercept=True,
        max_iter=1500,
        verbose=10,
        n_jobs=10,
        C=1
    )
    model = make_pipeline(MinMaxScaler(), lr)
    model.fit(x_train, y_train)
    joblib.dump(model, _model_dir / "lr_model.pkl")
    return model

def train_svm(x_train: List, y_train: List) -> Type[GridSearchCV]:
    # linear_svc= LinearSVC(
    #     penalty='l2',
    #     loss='squared_hinge',
    #     C=1,
    #     verbose=10,
    #     max_iter=1500,
    # )
    # svc = CalibratedClassifierCV(linear_svc)
    svc = SVC(
        kernel="sigmoid",
        # degree=4,
        C=1,
        gamma='scale',
        coef0=0,
        probability=True,
        cache_size=4000,
        verbose=True, max_iter=1500,
    )

    
    model = make_pipeline(StandardScaler(), svc)
    start_time = time.time()
    with joblib.parallel_backend('threading', n_jobs=5):
        model.fit(x_train, y_train)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    joblib.dump(model, _model_dir / "svm_model.pkl")
    return model


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
        model_type.title, file_name, y_test, y_prob
    )
    plots.save_confusion_matrix(model_type.title, file_name, y_test, y_pred)


def run() -> None:
    arguments = docopt(__doc__)

    x_train, x_test, y_train, y_test = get_test_train(get_massaged_df())
    if arguments["lr"]:
        name = "lr_best_model"
        model = train_lr(x_train, y_train)
        save_plots(ModelENUM.LR, name, x_test, y_test, model)
        stats = models.analyze_model(model, x_test, x_train, y_test, y_train)
        with open(_model_dir / f"{name}.json", 'w') as f:
            json.dump(stats, f)
    elif arguments["svm"]:
        name = "svm_best_model"
        model = train_svm(x_train, y_train)
        save_plots(ModelENUM.SVM, name, x_test, y_test, model)
        stats = models.analyze_model(model, x_test, x_train, y_test, y_train)
        with open(_model_dir / f"{name}.json", 'w') as f:
            json.dump(stats, f)
    elif arguments["ffn"]:
        model = train_ffn(x_train, y_train)
    

