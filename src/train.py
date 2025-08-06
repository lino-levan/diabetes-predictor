import math
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import DiabetesDataset
from neural_network import NeuralNetwork
from const import learning_rate, batch_size, epochs, loss_fn

# Create scaler and fit on training data
scaler = StandardScaler()
train = DiabetesDataset("data/train.csv", scaler=scaler, fit_scaler=True)
train_dataloader = DataLoader(train, batch_size=batch_size)

# Use same scaler for test data (no fitting)
test = DiabetesDataset("data/test.csv", scaler=scaler, fit_scaler=False)
test_dataloader = DataLoader(test, batch_size=batch_size)

model = NeuralNetwork()
print(model.parameters())

example_feature = train[0]["features"]
print(train[0]["label"])
print(example_feature)
print(model(example_feature))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train_loop(dataloader):
    model.train() # set model in training mode, only relevant for batch norm and dropout but /shrug
    size = len(dataloader.dataset)

    for batch, result in enumerate(dataloader):
        features = result["features"]
        labels = result["label"]

        # First get prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)

        # Now do the backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            print("Loss:", loss.item(), "[", batch, "/", math.ceil(size / batch_size), "]")

def test_loop(dataloader):
    model.eval() # set model in eval mode, only relevant for batch norm and dropout but /shrug
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # disables gradient tracking for perf improvement
        for result in dataloader:
            features = result["features"]
            labels = result["label"]

            pred = model(features)
            test_loss += loss_fn(pred, labels).item()

            pred_flat = pred.squeeze()
            label_flat = labels.squeeze()
            pred_classes = (torch.sigmoid(pred_flat) >= 0.5).float()
            correct += (pred_classes == label_flat).sum().item()

    print(correct, size)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader)
    test_loop(test_dataloader)

torch.save(model.state_dict(), "model_weights.pth")
