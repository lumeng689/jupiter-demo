from torch import nn
import torch
import jieba
import numpy as np

raw_text = """越努力就越幸运"""
words = list(jieba.cut(raw_text))
print(words)

word_to_ix = {i: word for i, word in enumerate(set(words))}
print(word_to_ix)

embeds = nn.Embedding(4,3)
print(embeds.weight[0])

keys = word_to_ix.keys()
keys_list = list(keys)

tensor_value = torch.LongTensor(keys_list)

print(embeds(tensor_value))
