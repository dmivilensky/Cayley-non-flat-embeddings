#!/usr/bin/env python3

import random
import numpy as np
import torch
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import Levenshtein as lev
from lr_scheduler import WarmupMultiStepLR


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 16)
        self.lstm = torch.nn.LSTM(
            16, 256, num_layers=2, bidirectional=False, dropout=0.2, batch_first=True
        )

    def forward(self, text, text_lengths):
        packed_embedded = pack_padded_sequence(
            self.embedding(text), text_lengths,
            batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return hidden


class GroupDataset(Dataset):
    def __init__(self, samples_count, generators):
        self.samples_count = samples_count
        self.generators = generators

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        length = max(1, int(np.random.poisson(lam=5.0)))
        sequence = [random.choice(
            list(range(1, 1 + 2 * self.generators))) for _ in range(length)]
        return {'sequence': (sequence, length)}


def collate_wrapper(batch):
    return pad_sequence([
        torch.IntTensor(sen['sequence'][0])
        for sen in batch
    ], batch_first=True, padding_value=0), [sen['sequence'][1] for sen in batch]


def pairwise_distances(sequences):
    batch_size = sequences.shape[0]
    result = np.zeros(shape=(batch_size, batch_size))
    for i in range(batch_size):
        for j in range(i + 1):
            result[i, j] = result[j, i] = lev.distance(
                "".join(map(str, sequences[i])), "".join(map(str, sequences[j])))
    return torch.Tensor(result)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupMultiStepLR(optimizer, warmup_iters=10)

criterion = torch.nn.MSELoss().to(device)

epochs = 40
batch_size = 10
steps = 30
samples_count = 100
generators = 2
group_dataset = GroupDataset(samples_count, generators)

for _ in range(epochs):
    epoch_loss = 0.0
    iters = 0

    model.train()
    data_loader = itertools.islice(DataLoader(
        group_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
        collate_fn=collate_wrapper
    ), batch_size*steps)

    for batch in data_loader:
        optimizer.zero_grad()
        sequences, sequences_lengths = batch

        embeddings = model(sequences, sequences_lengths)
        loss = criterion(
            torch.cdist(embeddings, embeddings, p=2).unsqueeze(0),
            pairwise_distances(sequences).unsqueeze(0)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        iters += 1

    print('loss:', epoch_loss / iters)
