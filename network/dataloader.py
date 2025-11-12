import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= self.num_samples:
            raise StopIteration

        end = self.cursor + self.batch_size
        batch_indices = self.indices[self.cursor:end]
        self.cursor = end

        batch = [self.dataset[i] for i in batch_indices]
        x, y = zip(*batch)
        return np.stack(x, axis=0), np.stack(y, axis=0)

