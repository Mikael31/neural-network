import numpy as np
import pytest
from network.dataset import Dataset

@pytest.fixture
def dataset():
    x = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    return Dataset(x, y)

def test_length(dataset):
    assert len(dataset) == 10

def test_getitem_returns_tuple(dataset):
    x, y = dataset[0]
    assert isinstance(x, np.ndarray)
    assert np.isscalar(y)
    assert x.shape == (2,)

def test_getitem_values_correct(dataset):
    x, y = dataset[3]
    expected_x = np.array([6, 7])
    expected_y = 3
    np.testing.assert_array_equal(x, expected_x)
    assert y == expected_y

def test_indexing_out_of_bounds(dataset):
    with pytest.raises(IndexError):
        dataset[100]

