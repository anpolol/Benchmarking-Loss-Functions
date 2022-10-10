from modules.negativeSampling import NegativeSampler
from torch_geometric.data import Data
import torch

data_normal = Data(edge_index = torch.tensor([[1,2,3,4,5,1,5,4,3,2],[2,3,4,5,1,5,4,3,2,1]]))

def test_normal_case():
    NS = NegativeSampler(data_normal)
    batch = torch.tensor([1,2,3,4,5])
    assert NS.negative_sampling(batch, 2).shape == (10,2)

def test_excess_neighbours_case():
    NS = NegativeSampler(data_normal)
    batch = torch.tensor([1,2,3,4,5])
    assert NS.negative_sampling(batch, 1).shape == (5,2)

def test_bigger_number_of_required_neighbours_case():
    NS = NegativeSampler(data_normal)
    batch = torch.tensor([1,2,3])
    assert NS.negative_sampling(batch, 3).shape == (2,2)

def test_not_connected_case():
    NS = NegativeSampler(data_normal)
    batch = torch.tensor([1,3,4])
    assert NS.negative_sampling(batch, 2).shape == (4,2)

