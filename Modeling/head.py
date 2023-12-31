"""
Created on Nov 17, 2023

@author: Mamoutou Fofana

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
  '''
  Head of self-attetention in parallel
  '''
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(num_embed, head_size, bias=False)
    self.query = nn.Linear(num_embed, head_size, bias=False)
    self.value = nn.Linear(num_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B, T, C = x.shape
    k = self.key(x) # (B, T, hs)
    q = self.query(x) # (B, T, hs)
    #compute attention scores ("affinities")
    weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) -> (B, hs, T) -> (B, T, T)
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    weight = F.softmax(weight, dim=-1) # (B, T, T)
    weight = self.dropout(weight)
    # perform the weighted aggregation of the values
    val = self.value(x) # (B, T, hs)
    output = weight @ val # (B, T, T) dot (B, T, hs) -> (B, T, hs)

    return output