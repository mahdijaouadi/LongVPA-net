import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
from sklearn.metrics import mean_absolute_error
import random
import matplotlib.pyplot as plt
import os
from model.model_architecture import *
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import random
from config.train_config import *
current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model=MyModel(layers=resnet_layers,hidden_lstm_size=hidden_lstm_size,num_lstm_layers=num_lstm_layers,output_size=output_size).to(device)
# model = model.half()
model.apply(init_weights)
model_path = os.path.join(current_dir, "..", "models", "model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path,weights_only=False))
best_model=MyModel(layers=resnet_layers,hidden_lstm_size=hidden_lstm_size,num_lstm_layers=num_lstm_layers,output_size=output_size)

best_model_path = os.path.join(current_dir, "..", "models", "best_model.pth")
if os.path.exists(best_model_path):
    best_model.load_state_dict(torch.load(best_model_path,weights_only=False))
else:
    best_model.load_state_dict(model.state_dict())


class MultiModalDiskDataset(Dataset):
    def __init__(self, index_file, transform=None):
        with open(index_file, "r") as f:
            self.entries = json.load(f)
        self.transform = transform

        # Preload targets for class balancing
        self.targets = []
        for entry in self.entries:
            target_tensor = torch.load(entry["target"])
            target_value = target_tensor.item()
            self.targets.append(target_value)

        # Assign class bins
        self.class_bins = {
            0: [],  # [-6, -3)
            1: [],  # [-3, 3)
            2: []   # [3, 6]
        }
        for idx, t in enumerate(self.targets):
            if t < -3:
                self.class_bins[0].append(idx)
            elif t < 3:
                self.class_bins[1].append(idx)
            else:
                self.class_bins[2].append(idx)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        chart_1d = torch.load(entry["chart_1d"],weights_only=False)
        chart_1mo = torch.load(entry["chart_1mo"],weights_only=False)
        spy_seq = torch.load(entry["spy_seq"],weights_only=False)
        target = torch.load(entry["target"],weights_only=False)

        if self.transform:
            chart_1d = self.transform(chart_1d)
            chart_1mo = self.transform(chart_1mo)

        return chart_1d.to(torch.float32), chart_1mo.to(torch.float32), spy_seq.unsqueeze(-1), target


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size == 32, "This sampler is hardcoded for 32-sample batches."

        self.class_bins = dataset.class_bins

    def __iter__(self):
        for cls in self.class_bins:
            random.shuffle(self.class_bins[cls])

        min_batches = min(
            len(self.class_bins[0]) // 11,
            len(self.class_bins[1]) // 10,
            len(self.class_bins[2]) // 11
        )

        for i in range(min_batches):
            batch_indices = (
                self.class_bins[0][i * 11: (i + 1) * 11] +
                self.class_bins[1][i * 10: (i + 1) * 10] +
                self.class_bins[2][i * 11: (i + 1) * 11]
            )
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return min(
            len(self.class_bins[0]) // 11,
            len(self.class_bins[1]) // 10,
            len(self.class_bins[2]) // 11
        )

train_dataset = MultiModalDiskDataset(index_file=os.path.join(current_dir, "..","..","data","train","index.json"))
train_sampler = BalancedBatchSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)

validation_dataset = MultiModalDiskDataset(index_file=os.path.join(current_dir, "..","..","data","validation","index.json"))
validation_sampler = BalancedBatchSampler(validation_dataset)
validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, num_workers=4)

def get_predictions(x,loader):
    predictions=torch.tensor([])
    true_labels=torch.tensor([])
    model.eval()
    for chart_1d, chart_1mo, spy_seq, target in loader:
        with torch.no_grad():
            y_hat=model(chart_1d.to(device),chart_1mo.to(device),spy_seq.to(device)).to('cpu')   
        predictions=torch.cat((predictions,y_hat),dim=0)
        true_labels=torch.cat((true_labels,target),dim=0)
        torch.cuda.empty_cache()
    model.train()
    return true_labels,predictions

criterion=nn.MSELoss()
val_mae_hist=[]
training_mae_hist=[]
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
best_mae=1000
for epoch in range(EPOCHS):
    print(epoch)

    true_labels,predictions=get_predictions(model,loader=validation_loader)
    val_mae=mean_absolute_error(true_labels.numpy(),predictions.numpy())
    val_mae_hist.append(val_mae)


    true_labels,predictions=get_predictions(model,loader=train_loader)
    training_mae=mean_absolute_error(true_labels.numpy(),predictions.numpy())
    training_mae_hist.append(training_mae)
    print("/////////////")
    print(training_mae,val_mae)
    print("/////////////")
    if val_mae<best_mae:
        best_mae=val_mae
        best_model.load_state_dict(model.state_dict())
        torch.save(best_model.state_dict(), best_model_path)

    
    model.train()
    for chart_1d, chart_1mo, spy_seq, target in train_loader:
        y_hat=model(chart_1d.to(device),chart_1mo.to(device),spy_seq.to(device)).to('cpu')
        loss = criterion(y_hat,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    if epoch==5:
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2,weight_decay=1e-4)

    torch.cuda.empty_cache()
    








