import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Global variable
block_size = 64
batch_size = 128
max_iters = 1000
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2
num_embed = 384
num_head = 4
num_layer = 4




def text_reader(file_name):
	"""
	Return the list of characters and the vocabulary.

	Parameters
	----------
	filename : str, filename
	Returns
	-------
	chars : list of characters
	vocab_size : list of vocabulary
	"""
	file = open(file_name, 'r', encoding='utf-8' )
	text = file.read()
	chars = chars = sorted(list(set(text)))
	print(chars)
	# Vocabulary size
	vocab_size = len(chars)
	return chars, vocab_size, text

def encode_lambda(chars):
	"""
	Lambda Encode function ( encode list of character into int)

	parameters:
	string_:
	return:
	encode: list of characters encoded into int
	"""
	string_to_int = { char:i for i,char in enumerate(chars) }
	encode = lambda s: [string_to_int[c] for c in s]
	return encode

def decode_lambda(chars):
	"""
	Lambda Decode function (decode list of int into string)

	parameters:
	chars:
	return:
	decode: list of int decoded into string
	"""
	int_to_string = { i:ch for i,ch in enumerate(chars) }
	decode = lambda l: ''.join([int_to_string[i] for i in l])

	return decode



def split_data(file):
	"""
	Split encoded text into train_data and val_data

	parameters:
	file: raw text file
	Returns
	train_data: train data
	val_data:  validation data
	"""

	chars, _ , text = text_reader(file)
	encode = encode_lambda(chars)
	data = torch.tensor(encode(text), dtype=torch.long)
	n = int(0.8*len(data))
	train_data = data[:n]
	val_data = data[n:]
	return train_data, val_data


file = '/content/drive/MyDrive/MOOCs/LLM/Agricultural-Science.txt'
train_data, val_data = split_data(file)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')
print('inputs:')
# print(x.shape)
print(x)
print('targets:')
print(y)


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch(split)
          logits, loss = model(X, Y)
          losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out
