import numpy as np
import pytest
from network.dataloader import DataLoader
from network.dataset import Dataset

@pytest.fixture
def simple_dataset():
    x = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    return Dataset(x, y)

def test_length_and_iteration(simple_dataset):
    loader = DataLoader(simple_dataset, batch_size=10, shuffle=False)
    batches = list(loader)
    assert len(batches) == 5

    x_all = np.concatenate([b[0] for b in batches], axis=0)
    y_all = np.concatenate([b[1] for b in batches], axis=0)
    assert np.array_equal(x_all, simple_dataset.x)
    assert np.array_equal(y_all, simple_dataset.y)

def test_shuffle_changes_order(simple_dataset):
    loader1 = DataLoader(simple_dataset, batch_size=10, shuffle=True)
    loader2 = DataLoader(simple_dataset, batch_size=10, shuffle=True)
    X1 = np.concatenate([b[0] for b in loader1], axis=0)
    X2 = np.concatenate([b[0] for b in loader2], axis=0)
    assert not np.array_equal(X1, X2)

def test_remaining_elements(simple_dataset):
    loader = DataLoader(simple_dataset, batch_size=13, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4
    last_X, last_y = batches[-1]
    assert last_X.shape[0] == 11
    assert last_y.shape[0] == 11

def test_batch_shapes(simple_dataset):
    loader = DataLoader(simple_dataset, batch_size=8, shuffle=False)
    for X_batch, y_batch in loader:
        assert isinstance(X_batch, np.ndarray)
        assert isinstance(y_batch, np.ndarray)
        assert X_batch.shape[1:] == (2,)
        assert y_batch.ndim == 1

def test_iter_resets_each_epoch(simple_dataset):
    loader = DataLoader(simple_dataset, batch_size=10, shuffle=True)
    first_epoch = list(loader)
    second_epoch = list(loader)
    assert len(first_epoch) == len(second_epoch)

