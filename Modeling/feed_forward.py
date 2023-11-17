"""
Created on Tues Nov  14 2023

@author: Mamoutou Fofana
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
  '''
  Linear layer followed by a non-linearity

  '''
  def __init__(self, num_embed):
     super().__init__()
     self.network = nn.Sequential(
        nn.Linear(num_embed, 4 * num_embed),
        nn.ReLU(),
        nn.Linear(4 * num_embed, num_embed),
        nn.Dropout(dropout),
     )

  def forward(self, x):
    return self.network(x)