import torch
import random
from model.model_architecture import *
from torch.utils.data import Dataset,Sampler
import json
import random
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