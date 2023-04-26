import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from pathlib import Path
import pickle
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, num_hidden_layers:int, num_nodes:int, num_features: int, dropout_prob:float, activation_fn) -> None:
        super(FeedForward, self).__init__()
        self.num_classes = 2
        self.activation = activation_fn
        dropout_prob = 0
        layers_list = [nn.Flatten(), nn.Linear(num_features, num_nodes)]
        if num_hidden_layers > 1:
            for i in range(num_hidden_layers):
                layers_list.extend([nn.Dropout(dropout_prob),self.activation(),nn.Linear(num_nodes, num_nodes)])

        layers_list.extend([nn.Dropout(dropout_prob),self.activation(), nn.Linear(num_nodes, 1)])
        self.layers= nn.Sequential(*layers_list)

    def forward(self, xb):
        xb = xb.to(self.layers[1].weight.dtype)
        return self.layers(xb)
        
    # TODO add momentum
    def fit(self, train_dataset: TensorDataset, validation_dataset: TensorDataset, batch_size: int, 
            epochs: int, loss_function, learning_rate: float, momentum:float, weight_decay:float):
        self.to(device)
        print("using: ","cuda" if torch.cuda.is_available() else "cpu")
        # create dataloader for batching, shuffle to avoid overfitting/batch correlation
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dl = DataLoader(validation_dataset, batch_size, shuffle=True)
        
        # TODO tune optimizer
        # opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=.9, weight_decay=.0000001) # create optimizer TODO weight decay
        # opt = torch.optim.SGD(self.parameters(), lr=learning_rate) # create optimizer TODO weight decay
        opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # store epoch losses
        training_losses = []
        validation_losses = []

        for epoch in tqdm(range(epochs)):
            self.train()
            epoch_training_loss = 0
            epoch_validation_loss = 0

            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                # run model on batch and get loss
                # predictions = torch.round(self(xb)).squeeze()
                # loss = loss_function(predictions, yb.to(torch.float32))
                predictions = self(xb).squeeze()
                loss = loss_function(predictions, yb.to(torch.float32))
    
                # Back Propagation
                loss.backward()  # compute gradient
                opt.step()       # update weights
                opt.zero_grad()  # reset gradient

            self.eval()
            with torch.no_grad():
                for xb_tr, yb_tr in train_dl:
                    xb_tr, yb_tr = xb_tr.to(device), yb_tr.to(device)
                    # train loss
                    train_predictions = self(xb_tr).squeeze()
                    # train_predictions = torch.round(self(xb_tr)).squeeze()
                    training_loss = loss_function(train_predictions, yb_tr.to(torch.float32))
                    epoch_training_loss += (training_loss * xb_tr.shape[0])

                for xb_val, yb_val in valid_dl:
                    xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                    # val loss
                    val_predictions = self(xb_val).squeeze()
                    # val_predictions = torch.round(self(xb_val)).squeeze()
                    validation_loss = loss_function(val_predictions, yb_val.to(torch.float32)) # get loss
                    epoch_validation_loss += (validation_loss * xb_val.shape[0])

                # get epoch loss
                num_train_samples = len(train_dataset)
                epoch_training_loss_normalized = epoch_training_loss / num_train_samples 
                training_losses.append(epoch_training_loss_normalized.item())

                num_val_samples = len(validation_dataset)
                epoch_validation_loss_normalized = epoch_validation_loss / (num_val_samples) 
                validation_losses.append(epoch_validation_loss_normalized.item())

        return training_losses, validation_losses

    def score(self, tensor_dataset: TensorDataset, batch_size:int=64):
        print("using: ","cuda" if torch.cuda.is_available() else "cpu")
        # reference: https://blog.paperspace.com/training-validation-and-accuracy-in-pytorch/
        self.eval()
        self.to(device)
        dataloader = DataLoader(tensor_dataset, batch_size, shuffle=True)
        all_predictions = torch.tensor([],dtype=torch.long)
        ground_truth_labels = torch.tensor([],dtype=torch.long)
        all_predictions, ground_truth_labels = all_predictions.to(device), ground_truth_labels.to(device)

        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                xb, yb = xb.to(device), yb.to(device)

                # run model on batch
                class_probabilities = self(xb)
            
                # choose most likely class for each sample
                predictions = torch.round(torch.sigmoid(class_probabilities)).squeeze()

                # create running tensor with all true_label, predicted_label pairs 
                ground_truth_labels = torch.cat((ground_truth_labels, yb.to(torch.float32)))
                all_predictions = torch.cat((all_predictions, predictions))
            
            # create a stack where the top layer is the ground truth, bottom is prediction
            true_pred_stack = torch.stack((ground_truth_labels, all_predictions))
            classifier_scores = classification_report(ground_truth_labels, all_predictions)
            confusion_mtx = confusion_matrix(ground_truth_labels, all_predictions)
            class_accuracy = {i:0 for i in range(self.num_classes)} # dictionary containing class accuracies
            for label in range(self.num_classes):
                # get all pairs with the same true label in the tensor stack
                class_pairs = list(filter(lambda pair: pair[0] == label, true_pred_stack.T)) 

                # calculate how many have correctly been predicted (true = predicted)
                num_correct = len(list(filter(lambda pair: pair[0] == pair[1], class_pairs)))

                # find accuracy
                class_accuracy[label] = num_correct/len(class_pairs)
                
            # get mean accuracy
            correct = sum(all_predictions == ground_truth_labels).item()
            mean_accuracy = correct / len(tensor_dataset)
            
            return mean_accuracy, class_accuracy, classifier_scores, confusion_mtx

    def predict(self, tensor_dataset: TensorDataset, batch_size:int=64):
        print("using: ","cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        dataloader = DataLoader(tensor_dataset, batch_size, shuffle=True)
        y_prob = torch.tensor([],dtype=torch.long)
        ground_truth_labels = torch.tensor([],dtype=torch.long)
        y_prob, ground_truth_labels = y_prob.to(device), ground_truth_labels.to(device)

        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                xb, yb = xb.to(device), yb.to(device)

                # run model on batch
                class_probabilities = self(xb)
            
                # choose most likely class for each sample
                probabilities = torch.sigmoid(class_probabilities).squeeze()

                # create running tensor with all true_label, predicted_label pairs 
                ground_truth_labels = torch.cat((ground_truth_labels, yb.to(torch.float32))).to(torch.int32)
                y_prob = torch.cat((y_prob, probabilities))
                y_pred = torch.round(y_prob).to(torch.int32)
            
            return ground_truth_labels, y_prob, y_pred



def plot_frequencies(title, data):
    
    fig, ax = plt.subplots(figsize=(6,3))
    bars = ax.bar(['delayed' if x==1 else 'on time' for x in data[0]], data[1])

    ax.set_title(f'Class Instances in {title.capitalize()} Dataset')
    ax.set_xlabel('Count')
    ax.set_ylabel('Labels')

    print(data[1])
    ax.bar_label(bars, data[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the chart
    plt.show()

def ffn_evaluate(model, train_ds, test_ds, valid_ds):
# plot losses
    plt.plot(model["training_losses"], label="Training Loss")
    # print(model["training_losses"])
    plt.plot(model["valid_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # calculate accuracy
    d = {"train": train_ds, "test": test_ds, "validation": valid_ds}
    for dataset in d:
        print(f"Evaluating **{dataset}** dataset:")
        mean_accuracy, class_accuracy, classifier_scores, confusion_matrix = model["model"].score(d[dataset], model["bs"])
        print(f"Mean Accuracy: {mean_accuracy*100:.3f}")
        print(f"Mean per-class accuracy:")
        for key in class_accuracy:
            print(f"  {'delayed' if key==1 else 'on time'}{': '}{class_accuracy[key]*100:.3f}%")
        print(classifier_scores)
        print(confusion_matrix)
        print()

def ffn_tune(param_dict, loss_function, num_features, train_ds, valid_ds):
    # best_model = {key:param_dict[key] for key in param_dict}
    best_model = {}
    best_model["best_loss"] = np.inf # infinity
    for num_nodes in param_dict["num_nodes"]:
        for num_layers in param_dict["num_layers"]:
            for bs in param_dict["bs"]:
                for epoch in param_dict["epoch"]:
                    for lr in param_dict["learning_rate"]:
                        for momentum in param_dict["momentum"]:
                            for weight_decay in param_dict["weight_decay"]:
                                for dropout_prob in param_dict["dropout_prob"]:
                                    # use validation loss
                                    model = FeedForward(num_hidden_layers=num_layers, 
                                                        num_nodes=num_nodes, 
                                                        num_features=num_features, 
                                                        dropout_prob=dropout_prob, 
                                                        activation_fn=param_dict["activation_fn"])
                                    training_losses, valid_losses = model.fit(train_dataset=train_ds, 
                                                                            validation_dataset=valid_ds, 
                                                                            batch_size=bs, 
                                                                            epochs=epoch, 
                                                                            loss_function=loss_function, 
                                                                            learning_rate=lr,
                                                                            momentum=momentum,
                                                                            weight_decay=weight_decay)
                                    if valid_losses[-1] < best_model["best_loss"]:
                                        best_model["model"]=model
                                        best_model["best_loss"] = valid_losses[-1]
                                        best_model["epoch"] = epoch
                                        best_model["learning_rate"] = lr
                                        best_model["bs"] = bs
                                        best_model["valid_losses"] = valid_losses
                                        best_model["training_losses"] = training_losses
                                        best_model["num_layers"] = num_layers
                                        best_model["num_nodes"] = num_nodes
                                        best_model["dropout_prob"] = dropout_prob
                                        best_model["weight_decay"] = weight_decay
                                        best_model["momentum"] = momentum
                                        best_model["activation_fn"] = param_dict["activation_fn"]

                                    print("best loss: ", best_model["best_loss"])
    return best_model

def run_model(param_dict:dict, train_ds, test_ds, valid_ds, num_features:int):
    loss_function = nn.BCEWithLogitsLoss()
    best_model = ffn_tune(param_dict, loss_function, num_features, train_ds=train_ds, valid_ds=valid_ds)
    print_table = [
        ["Best batch size:", best_model["bs"]],
        ["Best epoch:", best_model["epoch"]],
        ["Best learning rate:", best_model["learning_rate"]],
        ["Best num nodes:", best_model["num_nodes"]],
        ["Best num layers:", best_model["num_layers"]],
        ["Best momentum:", best_model["momentum"]],
        ["Best weight decay:", best_model["weight_decay"]],
        ["Best dropout probability:", best_model["dropout_prob"]]
        ]

    print(tabulate(print_table, headers=["Hyperparameter", "Value"], tablefmt="grid"))

    ffn_evaluate(model=best_model,
                 train_ds=train_ds,
                 test_ds=test_ds,
                 valid_ds=valid_ds
                 )
    
    state = best_model["model"].state_dict()
    best_model_params={"num_nodes": best_model["num_nodes"], 
                       "num_layers":best_model["num_layers"], 
                       "num_features":num_features,
                       "state":state,
                       "dropout_prob":best_model["dropout_prob"],
                       "activation_fn":param_dict["activation_fn"]}
    return best_model_params


def save_model_pkl(model_params:dict):
    num_layers = model_params["num_layers"]
    num_nodes = model_params["num_nodes"]
    filename = Path.cwd().parent / "models" / f"{num_layers}_{num_nodes}_state_dict.pkl"
    with open(filename, 'wb') as pkl_file:
        pickle.dump(deepcopy(model_params), pkl_file)

def load_model(filename):
    with open(filename, 'rb') as pkl_file:
        params = pickle.load(pkl_file)
    model = FeedForward(num_hidden_layers=params["num_layers"], 
                    num_nodes=params["num_nodes"], 
                    num_features=params["num_features"], 
                    dropout_prob=params["dropout_prob"], 
                    activation_fn=params["activation_fn"])
    model.load_state_dict(params["state"])
    return model
