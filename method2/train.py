# -*- coding:utf-8 -*-
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
import json

import sys
sys.path.append("..")

from MUSEAttention import MUSEAttention


# 定义用于消岐的LSTM网络
class DisambiguationLSTM(nn.Module):
    def __init__(self, n_word, word_dim, word_hidden, n_pronounce):
        super(DisambiguationLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim)
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=word_hidden, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(word_hidden * 2, n_pronounce)

    def forward(self, x):
        x = self.word_embedding(x)
        x, _ = self.lstm(x)
        x_out = x[:, -1, :]
        x = self.linear1(x_out)
        return x
    
def make_sequence(x, dic):
    idx = [dic[i] for i in x if i in dic]
    idx = Variable(torch.LongTensor(idx))
    return idx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, default='v4', help='v1 是llm构建+人工清洗的数据集，v2 是人民日报文本+pypinyin自动构建的数据集, v3是群聊 csv 提供的数据集, v4 是群聊 sent 和 lb 数据集,v4_l 是更大的数据集和网络结构')
    parser.add_argument('--input', type=str, default='data/nlp_test.docx', help='输入文件名')
    parser.add_argument('--output', type=str, default='data/nlp_test_output.docx', help='输出文件名')
    parser.add_argument("--plot_curve", action="store_false", help="是否绘制 loss 曲线")
    args = parser.parse_args()

    if args.scale == 'v1':
        train_data_file = 'data/train_data.json'
        model_file = 'data/disambiguation_models.pth'
    elif args.scale == 'v2':
        train_data_file = 'data/train_data_v2.json'      # 原来的名字是 train_data_big.json ， 改一改名字
        model_file = 'data/disambiguation_models_v2.pth' # 原来的名字是 disambiguation_models_big.pth ， 改一改名字
    elif args.scale == 'v3':
        train_data_file = 'data/train_data_v3.json'
        model_file = 'data/disambiguation_models_v3_2.pth'
    elif args.scale == 'v4':
        train_data_file = 'data/train_data_v4.json'
        model_file = 'data/disambiguation_models_v4.pth'
    # elif args.scale == 'v4_l':
    #     train_data_file = 'data/train_data_v4_large.json'
    #     model_file = 'data/disambiguation_models_v4_large.pth'
    else:
        raise ValueError('scale 参数只能是 v1 或 v2 或 v3 或 v4')
    
    word_hidden_size = 128 if args.scale != 'v4_l' else 512
    word_dim = 100 if args.scale != 'v4_l' else 250

    # 读取训练数据
    with open(train_data_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 将每个字和多音字的注音编码
    word_to_idx = {}
    pron_to_idx = {}
    for word, examples in train_data.items():
        # 添加字
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
        for example in examples:
            sentence, _, pron = example
            for ch in sentence:  # 确保句子中的每个字都被编码
                if ch not in word_to_idx:
                    word_to_idx[ch] = len(word_to_idx)
            if pron not in pron_to_idx:
                pron_to_idx[pron] = len(pron_to_idx)

    # 使用 nn.ModuleDict 打包每个字的LSTM网络
    models = nn.ModuleDict()
    for word in train_data.keys():  # 只为 train_data 中的键构建模型
        print(f'Building model for {word}')
        models[word] = DisambiguationLSTM(len(word_to_idx) + 1, 100, word_hidden_size, len(pron_to_idx))
    models = models.cuda()

    # 定义损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optimizers = {word: optim.Adam(models[word].parameters(), lr=0.01) for word in models.keys()}
    
    loss_list = []
    # 训练
    for epoch in range(100):
        print('*' * 10)
        print(f'epoch {epoch + 1}')
        for word, examples in train_data.items():
            running_loss = 0
            for example in examples:
                sentence, _, pron = example
                word_list = make_sequence(sentence, word_to_idx)
                word_list = word_list.cuda()  # 添加 batch 维度
                pron_list = [pron_to_idx[pron]]
                # 将 pron tensor 到 cuda 上
                pron_tensor = Variable(torch.LongTensor(pron_list))
                pron_tensor = pron_tensor.cuda()
                out = models[word](word_list.unsqueeze(0))  # 添加 batch 维度
                loss = loss_func(out, pron_tensor)
                running_loss += loss.item()
                optimizers[word].zero_grad()
                loss.backward()
                if word == "了":
                    loss_list.append(loss.item())
                optimizers[word].step()
            print(f'Loss for {word}: {running_loss / len(examples)}')

    print("训练完成")

    # 保存模型
    torch.save(models.state_dict(), model_file)
    print("模型已保存")

    # 绘制 loss 曲线
    if args.plot_curve:
        import matplotlib.pyplot as plt
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.grid()
        plt.savefig('assets/loss_curve.png')
        plt.show()
