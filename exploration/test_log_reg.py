import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split


def analyze_model(model, X_test, X_train, y_test, y_train):
    # TODO score vs score_samples
    test_accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    prf = precision_recall_fscore_support(y_test, y_pred, average='binary') # TODO different average values: micro macro binary weighted samples

    # print(f"Test accuracy: {test_accuracy}")
    # print(f"Train accuracy: {train_accuracy}")
    # print(f"Precision: {prf[0]}")
    # print(f"Recall: {prf[1]}")
    # print(f"F-Beta Score: {prf[2]}")
    # print(f"F1 Score: {f1}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

def read_csv_low_memory(csv):
    return pd.read_csv(csv, low_memory=False)



raw_dir = Path.cwd() / "raw"
file_pattern = 'Flights_2018_*.csv'
csv_files = glob.glob(str(raw_dir) + '/' + file_pattern)

print('Loaded Files')
print(csv_files)
df = pd.concat(map(read_csv_low_memory, csv_files))

df_modified = df.drop(columns=[
                               'Div1Airport', 
                               'Div1AirportID', 
                               'Div1AirportSeqID', 
                               'Div1WheelsOn', 
                               'Div1TotalGTime', 
                               'Div1LongestGTime', 
                               'Div1WheelsOff', 
                               'Div1TailNum',
                               'Div2Airport', 
                               'Div2AirportID', 
                               'Div2AirportSeqID', 
                               'Div2WheelsOn', 
                               'Div2TotalGTime', 
                               'Div2LongestGTime', 
                               'Div2WheelsOff', 
                               'Div2TailNum',
                               'Div3Airport', 
                               'Div3AirportID', 
                               'Div3AirportSeqID', 
                               'Div3WheelsOn', 
                               'Div3TotalGTime', 
                               'Div3LongestGTime', 
                               'Div3WheelsOff', 
                               'Div3TailNum',
                               'Div4Airport', 
                               'Div4AirportID', 
                               'Div4AirportSeqID', 
                               'Div4WheelsOn', 
                               'Div4TotalGTime', 
                               'Div4LongestGTime', 
                               'Div4WheelsOff', 
                               'Div4TailNum',
                               'Div5Airport', 
                               'Div5AirportID', 
                               'Div5AirportSeqID', 
                               'Div5WheelsOn', 
                               'Div5TotalGTime', 
                               'Div5LongestGTime', 
                               'Div5WheelsOff', 
                               'Div5TailNum', 
                               'Duplicate', 
                               'Unnamed: 119'])
df_modified = df_modified.dropna(subset=['ArrDel15'])
print(df_modified[df_modified['ArrDel15'].isnull()])


df = df[["DayofMonth","Month","Year", "Distance","DepDelay", "ArrDel15"]].dropna()# get rid of nan
print(df.ArrDel15.unique()) # find unique values

X = df[["DayofMonth","Month","Year", "Distance", "DepDelay"]]
y = df[["ArrDel15"]]
y = y.ArrDel15.ravel() # flatten
print(y.shape)

# split into train and 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=150) # TODO how to choose random state


# Logistic Regression model creation
log_reg = LogisticRegression()

param_grid = [
  {'solver':['liblinear'],'C':[.001, .01, .5, .1, 1, 5, 10], 'penalty':['l1','l2'], 'n_jobs':[1]},
  {'solver':['saga'],'C':[.001, .01, .5, .1, 1, 5, 10], 'penalty':['l2','elasticnet'], 'n_jobs':[10]} # l1 gives erros here
]

print("Running log_req")
search = GridSearchCV(log_reg, param_grid, scoring='f1', cv=5, verbose=0) 
logreg_model = search.fit(X_train, y_train)
print("finished")
