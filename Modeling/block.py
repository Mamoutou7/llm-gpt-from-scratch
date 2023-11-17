"""
Created on Thu Nov  16 17:20:39 2023

@author: Mamoutou Fofana
"""

import torch.nn as nn


from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
  """
  Transformer block : communication followed by computation
  """
  def __init__(self, num_embed, num_head):
    super().__init__()
    head_size = num_embed // num_head
    self.self_attention = MultiHeadAttention(num_head, head_size)
    self.feed_forward = FeedForward(num_embed)
    self.linear1 = nn.LayerNorm(num_embed)
    self.linear2 = nn.LayerNorm(num_embed)


  def forward(self, x):

    y = self.self_attention(x)
    print(y)
    x = self.linear1(x + y)
    y = self.feed_forward(x)
    x = self.linear2(x + y)

    return x
