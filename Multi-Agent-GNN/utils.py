# -*- coding: utf-8 -*-
"""

Utilities for training multiple-DRL agents using the PPO or A2C algorithms with GNN.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and D. González G.,
"Intelligent Resource Allocation in Joint Radar-Communication With Graph Neural Networks,"
in IEEE Transactions on Vehicular Technology, vol. 71, no. 10, pp. 11120-11135, Oct. 2022, doi: 10.1109/TVT.2022.3187377.


"""

import numpy as np
from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Callable, Union, Optional

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def save_itr_info(fname, itr, av_reward):
    with open(fname, "w") as f:
        f.write(f"Iteration: {itr}\n Average reward: {av_reward}\n")

def pathlength(path):
    return len(path["reward"])

    
def save_variables(model, model_file):
    """Save parameters of the NN 'model' to the file destination 'model_file'. """
    torch.save(model.state_dict(), model_file)

def load_variables(model, load_path):
#    model.cpu()
    model.load_state_dict(torch.load(load_path)) #, map_location=lambda storage, loc: storage))
    # model.eval() # TODO1- comment:  to set dropout and batch normalization layers to evaluation mode before running inference
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class RLDataset(Dataset):
    def __init__(self, obs, next_obs, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = obs
        self.next_data = next_obs
        # self._indices: Optional[Sequence] = None
    
    # def __getitem__(
    #     self, 
    #     idx: Union[int, np.integer, IndexType],
    #     ) -> Union['Dataset', Data]:
    #     r"""In case :obj:`idx` is of type integer, will return the data object
    #     at index :obj:`idx` (and transforms it in case :obj:`transform` is
    #     present).
    #     In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
    #     tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
    #     bool, will return a subset of the dataset at the specified indices."""
    #     if (isinstance(idx, (int, np.integer))
    #             or (isinstance(idx, Tensor) and idx.dim() == 0)
    #             or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

    #         data = self.get(self.indices()[idx])
    #         data = data if self.transform is None else self.transform(data)
    #         return data, idx

    #     else:
    #         return self.index_select(idx), idx
    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        data = self.data[idx]
        next_data = self.next_data[idx]
        
        return data, next_data, idx
        