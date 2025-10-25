# python3.7 pytorch_GPU 1.13.1+cu117
# -*- coding: UTF-8 -*-
# @IDE    :
# @Author : keymon
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import time



print(torch.version)
# Read Data

# 读取氨基酸序列文本
# 正负样本的训练集
"""
read_csv()：从指定路径中读取数据
参数：
    sep：指定了制表符作为数据的分隔符
    header：数据文件中没有列名或标题行
"""
# negative_training_set1_txt = pd.read_csv('dataset/negative_training_set1(1038).txt', sep="\t", header=None)
# print(f'读取的负样本训练集1数据：\n{negative_training_set1_txt}')
# negative_training_set2_txt = pd.read_csv('dataset/negative_training_set2(1038).txt', sep="\t", header=None)
# negative_training_set3_txt = pd.read_csv('dataset/negative_training_set3(1038).txt', sep="\t", header=None)
# negative_training_set4_txt = pd.read_csv('dataset/negative_training_set4(1038).txt', sep="\t", header=None)
# negative_training_set5_txt = pd.read_csv('dataset/negative_training_set5(1038).txt', sep="\t", header=None)
# positive_training_set_txt1 = pd.read_csv('dataset/positive_training_set1(1038).txt', sep="\t", header=None)
# positive_training_set_txt2 = pd.read_csv('dataset/positive_training_set2(1038).txt', sep="\t", header=None)
# # positive_training_set_txt3 = pd.read_csv('dataset/positive_training_set1_reverse(1038).txt', sep="\t", header=None)
# # positive_training_set_txt4 = pd.read_csv('dataset/positive_training_set2_reverse(1038).txt', sep="\t", header=None)
# # positive_training_set_txt3 = pd.read_csv('dataset/positive_training_set_mask5(1038).txt', sep="\t", header=None)
# # positive_training_set_txt4 = pd.read_csv('dataset/positive_training_set_mask7(1038).txt', sep="\t", header=None)
#
# # 正负样本的测试集
# negative_test_set_txt = pd.read_csv('testset/negative_test_set(260).txt', sep="\t", header=None)
# positive_test_set_txt = pd.read_csv('testset/positive_test_set(260).txt', sep="\t", header=None)
# # 正负样本的验证集
# independent_negative_set_txt = pd.read_csv('independent_negative_set(3033).txt', sep="\t", header=None)
# independent_positive_set_txt = pd.read_csv('independent_positive_set(1131).txt', sep="\t", header=None)
#
# def preprocess_data(data):
#     """
#     功能：将数据转化为三列的dataframe  name position sequences
#     参数:
#     data : 氨基酸序列文本（dataframe）
#     返回:
#     data3 : 列数为4的dataframe（索引列，name，position，sequence）
#     """
#     # any(axis=1) 表示对每一行进行逻辑或操作
#     data1 = data[data.isnull().any(axis=1)].reset_index()  # isnull()方法是判断数据中是否有缺省值，有则为True，否则为False
#     data2 = data.dropna().reset_index()  # 删除所有含有缺省值的行，即不包括缺省值的所有行
#     data3 = pd.concat([data1, data2], axis=1, sort=False, ignore_index=True)  # 重新设置索引，从0开始，并移除原始索引
#     data3.drop(columns=[0, 2, 3], inplace=True)  # 删除第0、2、3列，并且直接修改data3，不返回新的对象
#     data3.rename(index=str, columns={1: "name", 4: "position", 5: "sequence"}, inplace=True)  # 给列重新命名
#     return data3
#
#
# # 训练集
# negative_training_set1 = preprocess_data(negative_training_set1_txt)
# print(f'训练样本1dataframe：{negative_training_set1}')
# negative_training_set2 = preprocess_data(negative_training_set2_txt)
# negative_training_set3 = preprocess_data(negative_training_set3_txt)
# negative_training_set4 = preprocess_data(negative_training_set4_txt)
# negative_training_set5 = preprocess_data(negative_training_set5_txt)
# positive_training_set1 = preprocess_data(positive_training_set_txt1)
# positive_training_set2 = preprocess_data(positive_training_set_txt2)
# # positive_training_set3 = preprocess_data(positive_training_set_txt3)
# # positive_training_set4 = preprocess_data(positive_training_set_txt4)
#
# # 测试集
# negative_test_set = preprocess_data(negative_test_set_txt)
# positive_test_set = preprocess_data(positive_test_set_txt)
#
# # 验证集
# independent_negative_set = preprocess_data(independent_negative_set_txt)
# independent_positive_set = preprocess_data(independent_positive_set_txt)
#
# # Data Preprocessing - Sequence Extraction and Labeling
#
# # 将序列拆分成一个个的字母组成的二维列表并将同类数据拼接到一起
# negative_seq1 = np.array([list(word) for word in negative_training_set1.sequence.values])
# print(f'将序列拆分成字母列表：\n{negative_seq1}')
# negative_seq2 = np.array([list(word) for word in negative_training_set2.sequence.values])
# negative_seq3 = np.array([list(word) for word in negative_training_set3.sequence.values])
# negative_seq4 = np.array([list(word) for word in negative_training_set4.sequence.values])
# negative_seq5 = np.array([list(word) for word in negative_training_set5.sequence.values])
# # 负样本训练数据
# """
# concatenate()：用于连接多个数组
# 参数：
#     arrays：数组
#     axis：用于指定连接的轴，axis=0，也就是按行进行连接
# """
# negative_seq = np.concatenate((negative_seq1,
#                                negative_seq2,
#                                negative_seq3,
#                                negative_seq4,
#                                negative_seq5), axis=0)
# # negative_seqnew = np.concatenate((negative_seq1,
# #                                   negative_seq2), axis=0, out=None)
# # negative_seq = negative_seq1
#
# # 正样本训练数据
# positive_seq1 = np.array([list(word) for word in positive_training_set1.sequence.values])
# positive_seq2 = np.array([list(word) for word in positive_training_set2.sequence.values])
# # positive_seq3 = np.array([list(word) for word in positive_training_set3.sequence.values])
# # positive_seq4 = np.array([list(word) for word in positive_training_set4.sequence.values])
# positive_seq = np.concatenate((positive_seq1, positive_seq2), axis=0)
# # positive_seq = positive_seq1
# # 正负样本验证集
# negative_seq_val = np.array([list(word) for word in independent_negative_set.sequence.values])
# positive_seq_val = np.array([list(word) for word in independent_positive_set.sequence.values])
# # 正负样本测试集
# negative_seq_test = np.array([list(word) for word in negative_test_set.sequence.values])
# positive_seq_test = np.array([list(word) for word in positive_test_set.sequence.values])

# 给训练集和测试集的样本打上标签
"""
zeros()：用于创建一个元素全为零的数组，在这里表示负样本标签
参数：
    shape：指定要创建的数组的维度
    dtype：用于指定数据类型

ones()：用于创建一个元素全为一的数组，在这里表示正样本标签
参数：
    shape：指定要创建的数组的维度
    dtype：用于指定数据类型
"""
# negative_lab = np.zeros((negative_seq.shape[0],), dtype=int)
# # negative_lab = np.zeros((negative_seq1.shape[0],), dtype=int)
# positive_lab = np.ones((positive_seq.shape[0],), dtype=int)
# # negative_labnew = np.zeros((negative_seqnew.shape[0],), dtype=int)
# negative_lab_val = np.zeros((negative_seq_val.shape[0],), dtype=int)
# positive_lab_val = np.ones((positive_seq_val.shape[0],), dtype=int)
# negative_lab_test = np.zeros((negative_seq_test.shape[0],), dtype=int)
# print(f'负样本测试集标签：\n{negative_lab_test}')
# positive_lab_test = np.ones((positive_seq_test.shape[0],), dtype=int)

balanced_data = False
token = True
window = 19  # 3 to 19 odd number
# 正负样本训练集数据和标签
# if balanced_data:
#     df_train_pos = positive_seq
#     df_train_neg = negative_seqnew
#     df_lab_pos = positive_lab
#     df_lab_neg = negative_labnew
# else:
#     df_train_pos = positive_seq
#     df_train_neg = negative_seq
#     df_lab_pos = positive_lab
#     df_lab_neg = negative_lab

# 磷酸化数据集
negative_training_set_txt = pd.read_csv('dataset/ST_data/ST_negative_training_set_19(4308).txt', header=None)
positive_training_set_txt = pd.read_csv('dataset/ST_data/ST_positive_training_set_19(4308).txt', header=None)
negative_test_set_txt = pd.read_csv('dataset/ST_data/ST_negative_test_set_19(1079).txt', header=None)
positive_test_set_txt = pd.read_csv('dataset/ST_data/ST_positive_test_set_19(1079).txt', header=None)
print(negative_training_set_txt)

negative_training_set_txt.rename(index=str, columns={0: "sequence"}, inplace=True)  # 给列重新命名
positive_training_set_txt.rename(index=str, columns={0: "sequence"}, inplace=True)  # 给列重新命名
negative_test_set_txt.rename(index=str, columns={0: "sequence"}, inplace=True)  # 给列重新命名
positive_test_set_txt.rename(index=str, columns={0: "sequence"}, inplace=True)  # 给列重新命名
print(negative_training_set_txt)
negative_seq = np.array([list(word) for word in negative_training_set_txt.sequence.values])
positive_seq = np.array([list(word) for word in positive_training_set_txt.sequence.values])
negative_seq_test = np.array([list(word) for word in negative_test_set_txt.sequence.values])
positive_seq_test = np.array([list(word) for word in positive_test_set_txt.sequence.values])

print('aaa')
print(negative_seq.shape)
negative_lab = np.zeros((negative_seq.shape[0],), dtype=int)
print(negative_lab.shape)
positive_lab = np.ones((positive_seq.shape[0],), dtype=int)
negative_lab_test = np.zeros((negative_seq_test.shape[0],), dtype=int)
positive_lab_test = np.ones((positive_seq_test.shape[0],), dtype=int)

df_train_pos = positive_seq
df_train_neg = negative_seq
df_lab_pos = positive_lab
df_lab_neg = negative_lab

start_w = 9 - int(window / 2)
end_w = 9 + int(window / 2) + 1

# 将正负样本连接到一起组合成正常的训练集和标签
"""
concatenate()：用于连接多个数组
参数：
    arrays：数组
    axis：用于指定连接的轴，axis=0，也就是按行进行连接
"""
print(df_train_neg.shape)
print(df_train_pos.shape)
dataset_X = np.concatenate((df_train_pos, df_train_neg), axis=0)
print(dataset_X.shape)
dataset_Y = np.concatenate((df_lab_pos, df_lab_neg), axis=0)
# dataset_X_val = np.concatenate((positive_seq_val, negative_seq_val), axis=0)
# dataset_Y_val = np.concatenate((positive_lab_val, negative_lab_val), axis=0)
dataset_X_test = np.concatenate((positive_seq_test, negative_seq_test), axis=0)
dataset_Y_test = np.concatenate((positive_lab_test, negative_lab_test), axis=0)

# 平衡验证集
# negative_seq_val_eq = negative_seq_val[1:len(positive_seq_val) + 1]
# negative_lab_val_eq = negative_lab_val[1:len(positive_lab_val) + 1]
# dataset_X_val_eq = np.concatenate((positive_seq_val, negative_seq_val_eq), axis=0, out=None)
# dataset_Y_val_eq = np.concatenate((positive_lab_val, negative_lab_val_eq), axis=0, out=None)

# 获得初步的词向量矩阵，相当于embedding过程，将所有的序列转化成一个n*19维度的矩阵
# asam是要标记化的字符序列，20 种天然氨基酸的缩写
asam = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
# 创建一个字符级别的标记器对象
tokenizer = Tokenizer(char_level=True)
# 使用标记器对象对文本进行编码
tokenizer.fit_on_texts(asam)

#  训练集
dataset_X_token = []
for i in range(len(dataset_X)):
    # 将每个文本转换为字符级别的编码序列
    temp = tokenizer.texts_to_sequences(dataset_X[i])
    # 将temp附加在dataset_X_token的末尾，变成一个新的数组
    dataset_X_token = np.append(dataset_X_token, temp)

dataset_X_token = dataset_X_token - 1  # 对每一个元素减1，将数据从以1开始编码变为以0开始编码
# 改变dataset_X_token的形状，使其变成一个二维数组，其中每行包含了一个完整的序列编码（即长度为 19）
dataset_X_token = dataset_X_token.reshape(len(dataset_X), 19)
dataset_X_token = dataset_X_token[:, range(start_w, end_w)]

# 验证集
# dataset_X_token_val = []
# for i in range(len(dataset_X_val)):
#     temp = tokenizer.texts_to_sequences(dataset_X_val[i])
#     dataset_X_token_val = np.append(dataset_X_token_val, temp)
#
# dataset_X_token_val = dataset_X_token_val - 1
# dataset_X_token_val = dataset_X_token_val.reshape(len(dataset_X_val), 19)
# dataset_X_token_val = dataset_X_token_val[:, range(start_w, end_w)]
#
# # 平衡之后的验证集
# dataset_X_token_val_eq = []
# for i in range(len(dataset_X_val_eq)):
#     temp = tokenizer.texts_to_sequences(dataset_X_val_eq[i])
#     dataset_X_token_val_eq = np.append(dataset_X_token_val_eq, temp)
#
# dataset_X_token_val_eq = dataset_X_token_val_eq - 1
# dataset_X_token_val_eq = dataset_X_token_val_eq.reshape(len(dataset_X_val_eq), 19)
# dataset_X_token_val_eq = dataset_X_token_val_eq[:, range(start_w, end_w)]

# 测试集
dataset_X_token_test = []
for i in range(len(dataset_X_test)):
    temp = tokenizer.texts_to_sequences(dataset_X_test[i])
    dataset_X_token_test = np.append(dataset_X_token_test, temp)

dataset_X_token_test = dataset_X_token_test - 1
dataset_X_token_test = dataset_X_token_test.reshape(len(dataset_X_test), 19)
dataset_X_token_test = dataset_X_token_test[:, range(start_w, end_w)]

print('bbb')
# (8616, 19)
print(dataset_X_token.shape)

#  将数据随机打乱
"""
shuffle()：用于将给定的数据集按照同一顺序洗牌打乱
参数：
    arrays：数组
    random_state：用于设置随机种子，控制随机过程的可重复性
"""
X_train, y_train = shuffle(dataset_X_token, dataset_Y, random_state=13)
# X_val, y_val = shuffle(dataset_X_token_val, dataset_Y_val, random_state=13)
# X_val_eq, y_val_eq = shuffle(dataset_X_token_val_eq, dataset_Y_val_eq, random_state=13)
X_test, y_test = shuffle(dataset_X_token_test, dataset_Y_test, random_state=13)

# 将Numpy数组转换为PyTorch张量的形式，并指定数据类型为int64
# X_train_torch = torch.from_numpy(X_train.astype(np.int64)).to('cuda')
# X_val_torch = torch.from_numpy(X_val.astype(np.int64)).to('cuda')
# X_val_eq_torch = torch.from_numpy(X_val_eq.astype(np.int64)).to('cuda')
# X_test_torch = torch.from_numpy(X_test.astype(np.int64)).to('cuda')
#
# y_train_torch = torch.from_numpy(y_train.astype(np.int64)).to('cuda')
# y_val_torch = torch.from_numpy(y_val.astype(np.int64)).to('cuda')
# y_val_eq_torch = torch.from_numpy(y_val_eq.astype(np.int64)).to('cuda')
# y_test_torch = torch.from_numpy(y_test.astype(np.int64)).to('cuda')

# 将数据传输到gpu上面进行运算
X_train_torch = torch.from_numpy(X_train.astype(np.int64))
# X_val_torch = torch.from_numpy(X_val.astype(np.int64)).cuda()
# X_val_eq_torch = torch.from_numpy(X_val_eq.astype(np.int64)).cuda()
X_test_torch = torch.from_numpy(X_test.astype(np.int64))

y_train_torch = torch.from_numpy(y_train.astype(np.int64))
# y_val_torch = torch.from_numpy(y_val.astype(np.int64)).cuda()
# y_val_eq_torch = torch.from_numpy(y_val_eq.astype(np.int64)).cuda()
y_test_torch = torch.from_numpy(y_test.astype(np.int64))

# 超参数
num_epochs = 2  # 训练轮数，每个epoch是对整个数据集的一次完整遍历
learning_rate = 0.00001  # 学习率
weight_loss = torch.tensor([0.13, 0.87])
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，度量模型输出与真实标签之前的差异

encoder_number = 2  # 编码器数量
batch_size = 64  # 批次大小
# 将训练数据和标签打包成一个数据集对象
train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
# 训练数据加载器，将训练数据划分为多个批次，在训练过程中逐批次的加载数据
# 每一轮重新洗牌，随机性
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_attn_pad_mask(seq_q, seq_k):
    """
    用于生成多头注意力机制中的填充掩码
    :param seq_q: 输入张量Q，大小为(batch_size, sequence_length)
    :param seq_k: 输入张量K，大小为(batch_size, sequence_length)
    :return: (batch_size, len_q, len_k)
    """
    batchSize, len_q = seq_q.size()
    batchSize, len_k = seq_k.size()
    # seq_k.data.eq(0) 获取张量的底层数据
    # eq(0)比较运算符，将seq_k中的元素与零进行比较，若相等则为True,返回的大小与seq_k的大小相同，是一个布尔张量
    """
    在多头注意力计算中，我们可能会根据查询序列 seq_q 的长度来计算输出，因此我们需要将填充掩码
    的形状扩展到 (batch_size, len_q, len_k)，
    其中 len_q 表示查询序列的长度。
    """
    # print(batchSize)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batchSize, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点机注意力机制
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @staticmethod
    def forward(Q, K, V, attn_mask):
        """
        前向传播
        :param Q: 查询矩阵，形状为[batch_size x n_heads x len_q x d_q]
        :param K: 键矩阵，形状为[batch_size x n_heads x len_k x d_k]
        :param V: 值矩阵，形状为[batch_size x n_heads x len_k x d_v]
        :param attn_mask: 填充掩码，[batch_size x n_heads x len_q x len_k]
        :return:
            context: 张量，包含了根据注意力权重分配给值矩阵V的加权和，其中每个位置的值代表了该位置的上下文信息
            attn: 注意力权重
        """
        # 首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(512 / 9)
        # print(f'score shape: {scores.shape}')
        # 设置掩码，把被mask的地方置为无限小，也就是掩码矩阵中为True的地方，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9)
        # 将scores的每个元素通过指数函数除以所有元素的和来进行归一化，使得分数总和为1
        # dim=-1表示对最后一个维度进行softmax归一化计算
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化
        :param d_model: 输入维度
        :param dropout: 正则化
        :param max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建一个固定概率丢弃输入张量中的一些元素的dropout层
        self.dropout = nn.Dropout(p=dropout)
        # [max_len, d_model]形状的零填充张量，用于存储位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 用于生成一个从0到max_len-1的有序浮点数张量，并扩展为与pe相同的维度[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        """
        需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        """
        # 生成一个从0到d_model-1，增量为2的有序浮点数张量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 从0开始到最后面，步长为2，代表的是偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe形状是：[max_len, 1, d_model]，确保与注意力权重的维度保持一致
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 定义一个缓冲区，缓冲区不参与模型的训练，位置编码矩阵在训练过程中不会更新

    def forward(self, x):
        """
        x: 输入序列[seq_len, batch_size, d_model]
        """
        # 截取缓冲位置矩阵中与输入x大小相同的位置编码，并保留所有的维度
        x = x + self.pe[:x.size(0), :]
        # 减少过拟合，最终返回加了位置编码后并应用了dropout的张量x
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=19):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rotation_matrix = self.generate_rotation_matrix()
        self.positional_embedding = self.generate_positional_embedding()

    def generate_rotation_matrix(self):
        indices = torch.arange(self.d_model).unsqueeze(1)
        angles = torch.arange(self.d_model).unsqueeze(0) * torch.arange(self.d_model).unsqueeze(1) * 0.01
        return torch.cos(angles)

    def generate_positional_embedding(self):
        indices = torch.arange(self.max_seq_len).unsqueeze(1)
        angles = torch.arange(self.d_model).unsqueeze(0) * torch.arange(self.max_seq_len).unsqueeze(1) * 0.01
        return torch.cos(angles)

    def forward(self, x):
        self.positional_embedding = self.positional_embedding.to(x.device)
        self.rotation_matrix = self.rotation_matrix.to(x.device)

        x += self.positional_embedding.unsqueeze(0)
        x = torch.matmul(x, self.rotation_matrix)

        return x


class AAPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(AAPositionalEncoding, self).__init__()

        # 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        # 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        # 前十个位置编码
        pe = torch.zeros((max_len + 1) // 2, d_model)
        # aape = torch.zeros(max_len, d_model)
        # 生成9-0的pos序列
        position = torch.arange((max_len - 1) // 2, -1, -1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        print(len(pe[:, 0::2]))
        pe[:, 1::2] = torch.cos(position * div_term)  ##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        print(len(pe[:, 1::2]))
        # 上面代码获取之后得到的pe:[10*d_model]
        # 去掉最后一行位置编码（中间位点这行），剩9行
        pe1 = pe[:-1]
        # 倒过来并插入
        pe1 = pe1.flip(0)
        aape = torch.cat((pe, pe1), dim=0)

        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        aape = aape.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', aape)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是相等的，使用映射linear做一个映射得到参数矩阵Wq, Wk, Wv
        # 输入向量维度是512，每个线性层的输出维度为64，总共8个子头
        self.head = 9
        self.W_Q = nn.Linear(512, 64 * self.head)
        self.W_K = nn.Linear(512, 64 * self.head)
        self.W_V = nn.Linear(512, 64 * self.head)
        # 将维度为8*64映射回512维度，将8个子头的输出拼接
        self.linear = nn.Linear(self.head * 64, 512)
        # 对每个样本的512个特征向量进行标准化
        self.layer_norm = nn.LayerNorm(512)
        # self.rotary_pos = RotaryPositionalEmbedding(512)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, attn_mask):
        """
        前向传播
        :param Q: 查询矩阵，形状为[batch_size x len_q x d_model]
        :param K: 键矩阵，形状为[batch_size x len_k x d_model]
        :param V: 值矩阵，形状为[batch_size x len_k x d_model]
        :param attn_mask: 填充掩码，[batch_size x len_q x len_k]
        :return:
            output: [batch_size x len_q x d_model]
            attn: 注意力权重，[batch_size x n_heads x len_q x len_k]
        """
        residual, batchSize = Q, Q.size(0)

        # q = self.rotary_pos(Q)
        # k = self.rotary_pos(K)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 先映射，后分头
        q_s = self.W_Q(Q).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # q_s: [batch_size x n_heads x len_q x d_q]
        k_s = self.W_K(K).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # 输入进行的attn_mask形状是 batch_size x len_q x len_k
        # 得到新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个子头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1)

        # 进行缩放点积注意力机制，得到的结果有两个：
        # context: [batch_size x n_heads x len_q x d_v]
        # attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        # contiguous()保证转置后的张量是连续存储的
        context = context.transpose(1, 2).contiguous().view(batchSize, -1, self.head * 64)
        # 映射回[batch_size x len_q x d_model]
        output = self.linear(context)
        # output = self.dropout(output)
        return self.layer_norm(output + residual), attn, residual  # output: [batch_size x len_q x d_model]


class MultiHeadAttentionT(nn.Module):
    """
        多头注意力机制
        """

    def __init__(self):
        super(MultiHeadAttentionT, self).__init__()
        # 输入进来的QKV是相等的，使用映射linear做一个映射得到参数矩阵Wq, Wk, Wv
        # 输入向量维度是512，每个线性层的输出维度为64，总共8个子头
        self.head = 8
        self.W_Q = nn.Linear(19, 64 * self.head)
        self.W_K = nn.Linear(19, 64 * self.head)
        self.W_V = nn.Linear(19, 64 * self.head)
        # 将维度为8*64映射回512维度，将8个子头的输出拼接
        self.linear = nn.Linear(self.head * 64, 19)
        # 对每个样本的512个特征向量进行标准化
        self.layer_norm = nn.LayerNorm(19)
        # self.rotary_pos = RotaryPositionalEmbedding(512)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, attn_mask):
        """
        前向传播
        :param Q: 查询矩阵，形状为[batch_size x len_q x d_model]
        :param K: 键矩阵，形状为[batch_size x len_k x d_model]
        :param V: 值矩阵，形状为[batch_size x len_k x d_model]
        :param attn_mask: 填充掩码，[batch_size x len_q x len_k]
        :return:
            output: [batch_size x len_q x d_model]
            attn: 注意力权重，[batch_size x n_heads x len_q x len_k]
        """
        residual, batchSize = Q, Q.size(0)

        # q = self.rotary_pos(Q)
        # k = self.rotary_pos(K)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 先映射，后分头
        q_s = self.W_Q(Q).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # q_s: [batch_size x n_heads x len_q x d_q]
        k_s = self.W_K(K).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batchSize, -1, self.head, 64).transpose(1,
                                                                       2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # 输入进行的attn_mask形状是 batch_size x len_q x len_k
        # 得到新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个子头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1)

        # 进行缩放点积注意力机制，得到的结果有两个：
        # context: [batch_size x n_heads x len_q x d_v]
        # attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        # contiguous()保证转置后的张量是连续存储的
        context = context.transpose(1, 2).contiguous().view(batchSize, -1, self.head * 64)
        # 映射回[batch_size x len_q x d_model]
        output = self.linear(context)
        # output = self.dropout(output)
        return self.layer_norm(output + residual), attn, residual  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    """
    前馈神经网络
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 由两个一维卷积层(进行特征提取和编码)及一个归一化层(缓解神经网络训练过程中出现的梯度消失/爆炸问题)组成，
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        # 对每个样本的512个特征向量进行标准化
        self.layer_norm = nn.LayerNorm(512)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        :param inputs: 多头注意力机制的输出，[batch_size, len_q, d_model]
        :return:
            output: [batch_size x len_q x d_model]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # output = nn.SiLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        # output = self.dropout(output)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNetT(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNetT, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=19, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=19, kernel_size=1)
        self.layer_norm = nn.LayerNorm(19)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头注意力机制
        # self.enc_self_attnT = MultiHeadAttentionT()
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络
        # self.pos_ffnT = PoswiseFeedForwardNetT()

    def forward(self, enc_inputs, enc_self_attn_mask):
    # def forward(self, enc_inputs, enc_inputs_T, enc_self_attn_mask, enc_self_attn_mask_T):
        """
        前向传播
        :param enc_self_attn_mask_T:
        :param enc_inputs_T:
        :param enc_inputs: 加入位置编码后的输入，[batch_size, seq_len, d_model]
        :param enc_self_attn_mask: 填充掩码，(batch_size, len_q, len_k)
        :return:
            enc_outputs: 前馈神经网络的输出，[batch_size x len_q x d_model]
            attn: 注意力权重，[batch_size x n_heads x len_q x len_k]
        """
        enc_outputs, attn, multi_residual = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs_T, attn_T, multi_residual = self.enc_self_attnT(enc_inputs_T, enc_inputs_T, enc_inputs_T,
        #                                                             enc_self_attn_mask_T)
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs_T = self.pos_ffnT(enc_outputs_T)

        return enc_outputs, attn
        # return enc_outputs, enc_outputs_T, attn, attn_T


class Encoder(nn.Module):
    """
    编码器
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # 定义生成一个矩阵，大小是 src_vocab_size * d_model，将原序列由整数转换为向量表示
        self.src_emb = nn.Embedding(21, 512)
        # 位置编码情况，这里是固定的正余弦函数
        # self.pos_emb = PositionalEncoding(256)
        self.pos_emb = AAPositionalEncoding(512)
        # self.pos_emb = RelativePositionalEncoding(512)
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来
        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(encoder_number)])
        # 用于将编码器的输出维度512映射为2
        self.LinearL1 = nn.Linear(512, 2)

    def forward(self, enc_inputs):
        """

        :param enc_inputs: [batch_size x src_len]
        :return:
            enc_outputs: [batch_size, 2]
            enc_self_attns: [encoder_number, batch_size, len_q, d_model]
        """
        # 通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # 位置编码
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # 填充掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # print(f'enc_self_attn_mask shape:{enc_self_attn_mask.shape}')
        size = enc_inputs.shape[0]
        # enc_inputs_T_expanded = torch.zeros((size, 512))
        # print(enc_inputs.shape)
        # enc_inputs_T_expanded[:, :19] = enc_inputs
        # enc_self_attn_mask_T = get_attn_pad_mask(enc_inputs_T_expanded, enc_inputs_T_expanded)
        # print(f'enc_self_attn_mask_T shape:{enc_self_attn_mask_T.shape}')
        # 注意力权重矩阵
        enc_self_attns = []
        # enc_outputs_T = enc_outputs.transpose(1, 2)  # [batch_size, d_model, src_len]
        # 2层编码器
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # enc_outputs, enc_outputs_T, enc_self_attn, enc_self_attns_T = layer(enc_outputs, enc_outputs_T,
            #                                                                     enc_self_attn_mask,
            #                                                                     enc_self_attn_mask_T)
            enc_self_attns.append(enc_self_attn)
            # enc_self_attns.append(enc_self_attns_T)
        # permute()用于维度变换操作
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs_T = enc_outputs_T.permute(2, 0, 1)
        # enc_outputs_concat = torch.concat((enc_outputs, enc_outputs_T), dim=2)
        # enc_outputs: [batch_size x len_q x d_model]
        # 将输出结果转换在[0,1]范围内的概率，用于二分类问题或概率估计
        # 将最后一个时间步的特征进行线性变换和 Sigmoid 函数处理，该时间步的特征可以被视为整个序列的汇总信息或总结
        enc_outputs = torch.softmax(self.LinearL1(enc_outputs[enc_outputs.shape[0] - 1]), dim=-1)
        # enc_outputs = torch.sigmoid(self.LinearL1(enc_outputs_concat[enc_outputs_concat.shape[0] - 1]))
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # 编码层
        self.encoder = Encoder()

    def forward(self, enc_inputs):
        """

        :param enc_inputs: [batch_size, src_len]
        :return:
            enc_outputs: [batch_size, 2]

        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        return enc_outputs


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         # inputs: [B, 1] or [B], logits
#         # targets: [B], 0 or 1
#
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         probs = torch.sigmoid(inputs)
#         pt = torch.where(targets == 1, probs, 1 - probs)  # pt = p_t
#
#         focal_weight = (1 - pt) ** self.gamma
#         alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
#
#         loss = alpha_t * focal_weight * BCE_loss
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


# 创建Transformer的实例model
model = Transformer()
# 采用的Adam优化器更新模型的参数
# model.parameters()用于获取模型的所有可训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 获取最好的迭代次数
best_epoch = 0
best_loss = np.inf

# 训练模型
train_loss = []
train_acc = []
start = time.process_time()
for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    # 从数据加载器中加载训练数据和标签
    for X_train_torch, y_train_torch in train_data_loader:
        # 计算模型对当前输入的预测输出
        y_pred = model(X_train_torch)
        # 根据预测值和实际标签计算模型的损失值（交叉熵损失函数）
        # loss = FocalLoss(y_pred, y_train_torch)
        loss = criterion(y_pred, y_train_torch)
        train_loss.append(loss.item())
        # print(f'loss:{loss.item()}')
        # Backward pass and update
        # 反向求导，计算模型参数相对于损失的梯度
        loss.backward()
        # 更新模型参数，优化模型，使得模型能够在训练集上表现得更好，最小化损失函数
        optimizer.step()
        # 将模型参数的梯度值清零，以便下一个批次的训练使用，避免每次反向传播后，模型的参数梯度累加
        optimizer.zero_grad()
    acc_train = torch.max(y_pred, 1)[1].eq(y_train_torch).sum() / float(y_train_torch.shape[0])
    train_acc.append(acc_train.item())
    with open("./train_loss.txt", 'w') as train_loss_data:
        train_loss_data.write(str(train_loss))
    with open("./train_acc.txt", 'w') as train_acc_data:
        train_acc_data.write(str(train_acc))
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
    # save best epoch
    # loss.item()获取当前迭代步骤中的损失值
    # if loss.item() < np.inf:  # 检查损失值是否存在且不为无穷大，排除异常
    #     valid_best_loss = loss.item()
    #     best_epoch = epoch
    #     # 将当前模型的参数保存在文件中
    #     torch.save(model.state_dict(), "best_model.pth")
end = time.process_time()
print(f'Best Epoch: {best_epoch}')
print("Model Training Time:", end - start)

torch.cuda.empty_cache()
# validation
# 将加载的模型参数加载到当前的模型中
# model.load_state_dict(torch.load("best_model.pth"))
# 上下文管理器，临时关闭梯度计算从而减少内存消耗并提高性能
# with torch.no_grad():
#     start1 = time.process_time()
#     # 未平衡的验证集
#     y_predicted_imbalanced = model(X_val_torch)
#     end1 = time.process_time()
#     print('Model Imbalanced Validation Time:', end1 - start1)
#     start2 = time.process_time()
#     # 已平衡的验证集
#     y_predicted_balanced = model(X_val_eq_torch)
#     end2 = time.process_time()
#     print('Model Balances Validation Time:', end2 - start2)
#     # max()返回每一行的最大值以及对应的索引，1表示沿第一个维度进行操作
#     # eq()将最大值与真实标签相比然后求和，除以总的标签数
#     acc1 = torch.max(y_predicted_imbalanced, 1)[1].eq(y_val_torch).sum() / float(y_val_torch.shape[0])
#     acc2 = torch.max(y_predicted_balanced, 1)[1].eq(y_val_eq_torch).sum() / float(y_val_eq_torch.shape[0])
#
#     # F1（2*准确率*召回率/准确率+召回率）召回率=灵敏度
#     # 模型精确率和召回率的一种调和平均，它的最大值是1，最小值是0
#     f1_score_imbalance = f1_score(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), y_val_torch.cpu().numpy(),
#                                   average='macro')  # 宏平均
#     f1_score_balance = f1_score(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy(),
#                                 average='macro')
#     # MCC它的取值范围为[-1,1]，取值为1时表示对受试对象的完美预测
#     # 取值为0时表示预测的结果还不如随机预测的结果，-1是指预测分类和实际分类完全不一致
#     mcc_imbalance = matthews_corrcoef(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(),
#                                       y_val_torch.cpu().numpy())
#     mcc_balance = matthews_corrcoef(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy())
#     # 绘制ROC曲线，获得特异度和灵敏度（假正率和真正率）和用于计算特异度和灵敏度的阈值
#     # 特异度（预测准确的正样本/正样本） 灵敏度（预测准确的负样本/负样本）
#     # fpr, tpr, thresholds = metrics.roc_curve(y_val_torch.numpy(), torch.max(y_predicted_imbalanced, 1)[1].numpy(), pos_label=1)
#     # auc_imbalance = metrics.auc(fpr, tpr)
#     # fpr, tpr, thresholds = metrics.roc_curve(y_val_eq_torch.numpy(), torch.max(y_predicted_balanced, 1)[1].numpy(), pos_label=1)
#     # auc_balance = metrics.auc(fpr, tpr)
#     auroc_imbalance = roc_auc_score(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), y_val_torch.cpu().numpy())
#     auroc_balance = roc_auc_score(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy())
#
#     auprc_imbalance = average_precision_score(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(),
#                                               y_val_torch.cpu().numpy())
#     auprc_balance = average_precision_score(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(),
#                                             y_val_eq_torch.cpu().numpy())
#
#     confusion_matrix_imbalanced = confusion_matrix(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(),
#                                                    y_val_torch.cpu().numpy())
#     confusion_matrix_balanced = confusion_matrix(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(),
#                                                  y_val_eq_torch.cpu().numpy())
#
#     tn_imbalanced, fp_imbalanced, fn_imbalanced, tp_imbalanced = confusion_matrix_imbalanced.ravel()
#     specificity_imbalanced = tn_imbalanced / (tn_imbalanced + fp_imbalanced)
#     sensitivity_imbalanced = tp_imbalanced / (tp_imbalanced + fn_imbalanced)
#
#     tn_balanced, fp_balanced, fn_balanced, tp_balanced = confusion_matrix_balanced.ravel()
#     specificity_balanced = tn_balanced / (tn_balanced + fp_balanced)
#     sensitivity_balanced = tp_balanced / (tp_balanced + fn_balanced)
#
# # print(f'Validation accuracy (imbalance): {acc1:.4f}, F1: {f1_score_imbalance:.4f}, mcc: {mcc_imbalance:.4f}, auc: {auc_imbalance:.4f}')
# print(
#     f'Validation accuracy (imbalance): {acc1:.4f}, F1: {f1_score_imbalance:.4f}, mcc: {mcc_imbalance:.4f}, auroc: {auroc_imbalance:.4f}, auprc: {auprc_imbalance:.4f}')
# print(f'Validation confusion matrix (imbalance): {confusion_matrix_imbalanced}')
# print(f'Validation confusion matrix (balance): {confusion_matrix_balanced}')
# print(f'Validation(imbalance) specificity: {specificity_imbalanced:.4f}, sensitivity: {sensitivity_imbalanced:.4f}')
# print()
# # print(f'Validation accuracy (balance): {acc2:.4f}, F1: {f1_score_balance:.4f}, mcc: {mcc_balance:.4f}, auc: {auc_imbalance:.4f}')
# print(
#     f'Validation accuracy (balance): {acc2:.4f}, F1: {f1_score_balance:.4f}, mcc: {mcc_balance:.4f}, auroc: {auroc_balance:.4f}, auprc: {auprc_balance:.4f}')
# print(confusion_matrix(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy()))

model.load_state_dict(torch.load("best_model.pth"))
print(y_test_torch)
with torch.no_grad():
    start = time.process_time()
    y_predicted_test = model(X_test_torch)
    print(y_predicted_test)
    end = time.process_time()
    print("Model Test Time:", end - start)
    # 准确率（预测准确的正样本+预测准确的负样本/总样本）
    acc_test = torch.max(y_predicted_test, 1)[1].eq(y_test_torch).sum() / float(y_test_torch.shape[0])
    # F1（2*准确率*召回率/准确率+召回率）召回率=灵敏度
    # 模型精确率和召回率的一种调和平均，它的最大值是1，最小值是0
    f1_score_test = f1_score(torch.max(y_predicted_test, 1)[1].cpu().numpy(), y_test_torch.cpu().numpy(),
                             average='macro')
    # MCC它的取值范围为[-1,1]，取值为1时表示对受试对象的完美预测，取值为0时表示预测的结果还不如随机预测的结果，-1是指预测分类和实际分类完全不一致
    mcc_test = matthews_corrcoef(torch.max(y_predicted_test, 1)[1].cpu().numpy(), y_test_torch.cpu().numpy())
    # 绘制ROC曲线，获得特异度和灵敏度（假正率和真正率）和用于计算特异度和灵敏度的阈值
    # 特异度（预测准确的正样本/正样本） 灵敏度（预测准确的负样本/负样本）
    # fpr, tpr, thresholds = metrics.roc_curve(y_test_torch.numpy(), torch.max(y_predicted_test, 1)[1].numpy(), pos_label=1)
    # 通过特异度和灵敏度作为参数得出AUC指标（ROC曲线下面积）
    # auc_test = metrics.auc(fpr, tpr)
    auroc_test = roc_auc_score(torch.max(y_predicted_test, 1)[1].cpu().numpy(), y_test_torch.cpu().numpy())
    auprc_test = average_precision_score(torch.max(y_predicted_test, 1)[1].cpu().numpy(), y_test_torch.cpu().numpy())

    confusion_matrix_test = confusion_matrix(torch.max(y_predicted_test, 1)[1].cpu().numpy(),
                                             y_test_torch.cpu().numpy())

    tn_test, fp_test, fn_test, tp_test = confusion_matrix_test.ravel()
    specificity_test = tn_test / (tn_test + fp_test)
    sensitivity_test = tp_test / (tp_test + fn_test)

print(
    f'Test accuracy: {acc_test:.4f}, F1: {f1_score_test:.4f}, mcc: {mcc_test:.4f}, auroc: {auroc_test:.4f}, auprc: {auprc_test:.4f}')
print(f'Test confusion matrix: {confusion_matrix_test}')
print(f'Test specificity: {specificity_test:.4f}, sensitivity: {sensitivity_test:.4f}')
print()
