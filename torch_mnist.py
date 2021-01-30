from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np
from IPython import embed
import torch
import math

# receive data
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
	print(PATH)
	content = requests.get(URL + FILENAME).content
	(PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
	((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)
# plt.show()
# embed()

# convert data into torch
x_train, y_train, x_valid, y_valid = map(
	torch.tensor, (x_train, y_train, x_valid, y_valid)
)

# print(x_train, y_train)
# print(x_train.shape) # torch.Size([50000, 784])
# print(y_train.min(), y_train.max())
# print(y_valid.shape) # torch.Size([10000])

# 1. Create a neural net from scratch
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad = True)

def log_softmax(x):
	return x - x.exp().sum(-1).log().unsqueeze(-1) # softmax normalization

def model(xb):
	return log_softmax(xb @ weights + bias) # @ stands for product operation

# 1. 1) create mini-batches and run
bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions

# print(preds[0], preds.shape) # preds.shape = torch.Size([64, 10])

# 1. 2) defining loss func.
def nll(input, target):
	# embed()
	return -input[range(target.shape[0]), target].mean()
	"""
	The input/targets are made of batches - the nll func here adds up and get the mean of each softmax loss.
	tensors can be used as indices. For example, input[0, 5] return tensor(-2.4278, grad_fn=<SelectBackward>)                         
	while input[0, [5]] returns tensor([-2.4278], grad_fn=<IndexBackward>) as output. 
	"""

loss_func = nll

yb = y_train[0:bs]
# print(loss_func(preds, yb))

# 1.3) defining accuracy: if the index with the largest value matches the target value, then the prediction was correct.
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


# 1.4) Make a training loop
from IPython.core.debugger import set_trace

lr = 0.002  # learning rate
epochs = 150  # how many epochs to train for

# for plotting
idx = []
acc = []
loss_plot = []
for epoch in range(epochs):
    for i in range((len(x_train) - 1) // bs + 1):
        #         set_trace()
        # embed()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

    # print(f"{epoch} epoch / accuracy: {accuracy(pred, yb)} / loss: {loss_func(pred, yb)}")

    idx.append(epoch)
    acc.append(accuracy(pred, yb))
    loss_plot.append(loss_func(pred, yb))

# for plotting accuracy and loss 

fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
host.set_ylim(0, 1)
par1.set_ylim(0, 2)

host.plot(idx, acc, "b-")
par1.plot(idx, loss_plot, "g-")

plt.show()

# 2. Refactoring

# Loss functions can be defined as 
import torch.nn.functional as F
loss_func_2 = F.cross_entropy

# creating model with nn.Module
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

# using model.parameters for learning
model = Mnist_Logistic()
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

# using nn.Linear
class Mnist_Logistic_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

# using optimizer
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

# 3. Using Dataset and DataLoader

# 3.1) Using Dataset
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        # instead of
        """
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        """
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# 3.2) Using Dataloader
# Dataloader gives us each minibatch automatically

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for epoch in range(epochs):
    for xb, yb in train_dl:
    	"""
    	instead of
    	for i in range((n-1)//bs + 1):
		    xb,yb = train_ds[i*bs : i*bs+bs]
		    pred = model(xb)
    	"""
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))


# 4. Adding validation
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
	"""
	Why two times more for validation datasets?
	We’ll use a batch size for the validation set that is twice as large as that for the training set. 
	This is because the validation set does not need backpropagation and thus takes less memory (it doesn’t need to store the gradients). 
	We take advantage of this to use a larger batch size and compute the loss more quickly.
	"""

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) # Checking validation loss

    print(epoch, valid_loss / len(valid_dl))

# 4.1) Refactoring 

# loss of each batch
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# fit()
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train() # into training mode
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval() # into evaluation mode
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl] #opt is none here
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

# refactoring get_data() func
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# all-in-one
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


# 5. Switching to CNN

# 5.1) Inheriting nn.Module
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4) # average pooling at the end
        return xb.view(-1, xb.size(1))

lr = 0.1
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# 5.2) Using nn.Sequential and Lambda

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

	def preprocess(x):
	    return x.view(-1, 1, 28, 28)

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)













