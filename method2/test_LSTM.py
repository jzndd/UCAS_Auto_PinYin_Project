# -*- coding:utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable

train_data = [
    ('行人走在路上', '行', 'xíng'),
    ('银行开门了', '行', 'háng'),
    ('他品行端正', '行', 'xíng'),
    ('这很行得通', '行', 'xíng'),
    ('这银行', '行', 'háng')
]

test_data = [
    ("行走", "行", "xíng"),
    ("银行", "行", "háng"),
]

# 将每个字和多音字的注音编码
word_to_idx = {}
pron_to_idx = {}
for words, _, prons in train_data:
    for ch in words:
        if ch not in word_to_idx:
            word_to_idx[ch] = len(word_to_idx)
    if prons not in pron_to_idx:
        pron_to_idx[prons] = len(pron_to_idx)
print(pron_to_idx)
print(word_to_idx)


# 用于消岐的神经网络
class DisambiguationLSTM(nn.Module):
    def __init__(self, n_word, word_dim, word_hidden, n_pronounce):
        super(DisambiguationLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim)
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=word_hidden, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(word_hidden*2, n_pronounce)

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.unsqueeze(0)      # x.size() : (1,5,100)
        x, _ = self.lstm(x)     # x.size(): (1,5,256)
        x_out = x[:, -1, :]     # 全连接层的输入为神经网络最后一层最后一个time step的输出
        x = self.linear1(x_out)     # x.size() : (1,2)
        return x


# loss函数和优化器
model = DisambiguationLSTM(len(word_to_idx) + 1, 100, 128, len(pron_to_idx))
print(model)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx


# 训练
for epoch in range(50):
    print('*' * 10)
    print('eopch{}'.format(epoch + 1))
    running_loss = 0
    for data in train_data:
        word, _, pron = data
        word_list = make_sequence(word, word_to_idx)
        pron_list = [pron_to_idx[pron]]
        pron = Variable(torch.LongTensor(pron_list))
        out = model(word_list)
        loss = loss_func(out, pron)
        running_loss += loss.data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {}'.format(running_loss / len(data)))
print()

# 测试
for w, _, p in test_data:
    input_seq = make_sequence(w, word_to_idx)
    test_out = model(input_seq)
    pred_y = torch.max(test_out, 1)[1].data.numpy()     # torch.max(a, 1) 返回每一行中的最大值，且返回其列索引
    print(list(pron_to_idx.keys())[list(pron_to_idx.values()).index(pred_y[0])], 'prediction')
    print(p, 'real')
