import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DiabetesDataset
from neural_network import NeuralNetwork

learning_rate = 1e-4
batch_size = 64
epochs = 10

test = DiabetesDataset("data/train.csv")
test_dataloader = DataLoader(test, batch_size=batch_size)
train = DiabetesDataset("data/train.csv")
train_dataloader = DataLoader(train, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using", device)

model = NeuralNetwork()
print(model.parameters())

example_feature = train[0]["features"]
print(train[0]["label"])
print(example_feature)
print(model(example_feature))

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train_loop(dataloader):
    model.train() # set model in training mode, only relevant for batch norm and dropout but /shrug
    size = len(dataloader.dataset)

    for batch, result in enumerate(dataloader):
        # First get prediction and loss
        pred = model(result["features"])
        loss = loss_fn(pred, result["label"])

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
            pred = model(result["features"])
            test_loss += loss_fn(pred, result["label"]).item()
            # For regression, calculate accuracy as percentage within threshold
            threshold = 0.5
            pred_flat = pred.squeeze()
            label_flat = result["label"].squeeze()
            within_threshold = (torch.abs(pred_flat - label_flat) <= threshold).type(torch.float)
            correct += within_threshold.sum().item()

    print(correct, size)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader)
    test_loop(test_dataloader)
