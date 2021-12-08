import json
import torch
import torch.nn as nn
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("intents.json", 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for i in intents["intents"]:
    tag = i["tag"]
    tags.append(tag)
    for p in i["patterns"]:
        w = tokenize(p)
        all_words.extend(w)
        xy.append((w, tag))

stop_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in stop_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (xi, yi) in xy:
    label = tags.index(yi)
    x_train.append(bag_of_words(xi, all_words))
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 10
input_size = len(all_words)     # len(x_train[0]) also works
hidden_size = 10
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1200

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for (w, labels) in train_loader:
        # forward
        outputs = model(w)
        loss = criterion(outputs, labels)

        # backward and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (e+1) % 100 == 0:
        print(f'epoch {e+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'Training complete. File save to {FILE}')

