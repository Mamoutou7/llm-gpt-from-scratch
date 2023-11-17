"""
Created on Nov 17, 2023

@author: Mamoutou Fofana

"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
  '''
  Multiple heads of self-attetention in parallel
  '''
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, num_embed)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    output = torch.cat([h(x) for h in self.heads], dim=-1) #(B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
    output = self.dropout(self.proj(output))
    return output