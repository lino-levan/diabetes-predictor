import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import DiabetesDataset
from neural_network import NeuralNetwork
from const import batch_size, loss_fn

model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

# Create scaler and fit on training data
scaler = StandardScaler()
train = DiabetesDataset("data/train.csv", scaler=scaler, fit_scaler=True)
train_dataloader = DataLoader(train, batch_size=batch_size)

# Use same scaler for test data (no fitting)
validation = DiabetesDataset("data/validation.csv", scaler=scaler, fit_scaler=False)
validation_dataloader = DataLoader(validation, batch_size=batch_size)

model.eval() # set model in eval mode, only relevant for batch norm and dropout but /shrug
size = len(validation)
num_batches = len(validation_dataloader)
validation_loss, correct = 0, 0

with torch.no_grad(): # disables gradient tracking for perf improvement
    for result in validation_dataloader:
        features = result["features"]
        labels = result["label"]

        pred = model(features)
        validation_loss += loss_fn(pred, labels).item()

        pred_flat = pred.squeeze()
        label_flat = labels.squeeze()
        pred_classes = (torch.sigmoid(pred_flat) >= 0.5).float()
        correct += (pred_classes == label_flat).sum().item()

print(correct, size)
validation_loss /= num_batches
correct /= size
print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validation_loss:>8f} \n")
