import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

class FeedForward(nn.Module):
    def __init__(self, num_hidden_layers, num_nodes:int, num_features: int) -> None:
        super(FeedForward, self).__init__()
        self.num_classes = 2
        if num_hidden_layers == 1:
             self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, 1),
            )
        else:
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, 1),
            )

    def forward(self, xb):
        xb = xb.to(self.layers[1].weight.dtype)
        return self.layers(xb)
        
    # TODO add momentum
    def fit(self, train_dataset: TensorDataset, validation_dataset: TensorDataset, batch_size: int, 
            epochs: int, loss_function, learning_rate: float):
        
        # create dataloader for batching, shuffle to avoid overfitting/batch correlation
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dl = DataLoader(validation_dataset, batch_size, shuffle=True)
        
        # TODO tune optimizer
        # opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=.9, weight_decay=.0000001) # create optimizer TODO weight decay
        # opt = torch.optim.SGD(self.parameters(), lr=learning_rate) # create optimizer TODO weight decay
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # store epoch losses
        training_losses = []
        validation_losses = []

        for epoch in tqdm(range(epochs)):
            self.train()
            epoch_training_loss = 0
            epoch_validation_loss = 0

            for xb, yb in train_dl:
                # run model on batch and get loss
                predictions = self(xb).squeeze()
                loss = loss_function(predictions, yb)
                # Back Propagation
                loss.backward()  # compute gradient
                opt.step()       # update weights
                opt.zero_grad()  # reset gradient

            self.eval()
            with torch.no_grad():
                for xb_tr, yb_tr in train_dl:
                    # train loss
                    train_predictions = self(xb_tr).squeeze()
                    training_loss = loss_function(train_predictions, yb_tr)
                    epoch_training_loss += (training_loss * xb_tr.shape[0])
                for xb_val, yb_val in valid_dl:
                    # val loss
                    val_predictions = self(xb_val).squeeze()
                    validation_loss = loss_function(val_predictions, yb_val) # get loss
                    epoch_validation_loss += (validation_loss * xb_val.shape[0])
                # get epoch loss
                num_train_samples = len(train_dataset)
                epoch_training_loss_normalized = epoch_training_loss / num_train_samples 
                training_losses.append(epoch_training_loss_normalized.item())

                num_val_samples = len(validation_dataset)
                epoch_validation_loss_normalized = epoch_validation_loss / (num_val_samples) 
                validation_losses.append(epoch_validation_loss_normalized.item())

        return training_losses, validation_losses

    def score(self, tensor_dataset: TensorDataset, batch_size:int=1):
        # reference: https://blog.paperspace.com/training-validation-and-accuracy-in-pytorch/
        self.eval()
        dataloader = DataLoader(tensor_dataset, batch_size, shuffle=True)
        all_predictions = torch.tensor([],dtype=torch.long)
        ground_truth_labels = torch.tensor([],dtype=torch.long)
        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                # run model on batch
                class_probabilities = self(xb)
                
                # choose most likely class for each sample
                predictions = (class_probabilities > 0.5).long().squeeze()
                
                # create running tensor with all true_label, predicted_label pairs 
                ground_truth_labels = torch.cat((ground_truth_labels, yb))
                all_predictions = torch.cat((all_predictions, predictions))
            
            # create a stack where the top layer is the ground truth, bottom is prediction
            true_pred_stack = torch.stack((ground_truth_labels, all_predictions))
            classifier_scores = precision_recall_fscore_support(ground_truth_labels, all_predictions)
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
                