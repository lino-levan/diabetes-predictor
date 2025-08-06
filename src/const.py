from torch import nn

learning_rate = 1e-4
batch_size = 64
epochs = 25

loss_fn = nn.BCEWithLogitsLoss()
