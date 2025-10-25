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
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.benchmark = False

# Read Data

# read txt data
negative_training_set1_txt = pd.read_csv('dataset/negative_training_set1(1038).txt', sep="\t", header=None)
negative_training_set2_txt = pd.read_csv('dataset/negative_training_set2(1038).txt', sep="\t", header=None)
negative_training_set3_txt = pd.read_csv('dataset/negative_training_set3(1038).txt', sep="\t", header=None)
negative_training_set4_txt = pd.read_csv('dataset/negative_training_set4(1038).txt', sep="\t", header=None)
negative_training_set5_txt = pd.read_csv('dataset/negative_training_set5(1038).txt', sep="\t", header=None)
positive_training_set_txt1 = pd.read_csv('dataset/positive_training_set1(1038).txt', sep="\t", header=None)
positive_training_set_txt2 = pd.read_csv('dataset/positive_training_set2(1038).txt', sep="\t", header=None)

negative_test_set_txt = pd.read_csv('testset/negative_test_set(260).txt', sep="\t", header=None)
positive_test_set_txt = pd.read_csv('testset/positive_test_set(260).txt', sep="\t", header=None)

independent_negative_set_txt = pd.read_csv('independent_negative_set(3033).txt', sep="\t", header=None)
independent_positive_set_txt = pd.read_csv('independent_positive_set(1131).txt', sep="\t", header=None)


# 将数据转化为三列的dataframe  name position sequences

#
def preprocess_data(data):
    data1 = data[data.isnull().any(axis=1)].reset_index()
    data2 = data.dropna().reset_index()
    data3 = pd.concat([data1, data2], axis=1, sort=False, ignore_index=True)
    data3.drop(columns=[0, 2, 3], inplace=True)
    data3.rename(index=str, columns={1: "name", 4: "position", 5: "sequence"}, inplace=True)
    return data3


negative_training_set1 = preprocess_data(negative_training_set1_txt)
negative_training_set2 = preprocess_data(negative_training_set2_txt)
negative_training_set3 = preprocess_data(negative_training_set3_txt)
negative_training_set4 = preprocess_data(negative_training_set4_txt)
negative_training_set5 = preprocess_data(negative_training_set5_txt)
positive_training_set1 = preprocess_data(positive_training_set_txt1)
positive_training_set2 = preprocess_data(positive_training_set_txt2)

negative_test_set = preprocess_data(negative_test_set_txt)
positive_test_set = preprocess_data(positive_test_set_txt)

independent_negative_set = preprocess_data(independent_negative_set_txt)
independent_positive_set = preprocess_data(independent_positive_set_txt)

# Data Preprocessing - Sequence Extraction and Labeling

# 将序列拆分成一个个的字母组成的二维列表并将同类数据拼接到一起
negative_seq1 = np.array([list(word) for word in negative_training_set1.sequence.values])
negative_seq2 = np.array([list(word) for word in negative_training_set2.sequence.values])
negative_seq3 = np.array([list(word) for word in negative_training_set3.sequence.values])
negative_seq4 = np.array([list(word) for word in negative_training_set4.sequence.values])
negative_seq5 = np.array([list(word) for word in negative_training_set5.sequence.values])
negative_seq = np.concatenate((negative_seq1,
                               negative_seq2,
                               negative_seq3,
                               negative_seq4,
                               negative_seq5), axis=0, out=None)
negative_seqnew = np.concatenate((negative_seq1,
                                  negative_seq2), axis=0, out=None)
positive_seq1 = np.array([list(word) for word in positive_training_set1.sequence.values])
positive_seq2 = np.array([list(word) for word in positive_training_set2.sequence.values])
positive_seq = np.concatenate((positive_seq1,
                               positive_seq2,), axis=0, out=None)

negative_seq_val = np.array([list(word) for word in independent_negative_set.sequence.values])
positive_seq_val = np.array([list(word) for word in independent_positive_set.sequence.values])

negative_seq_test = np.array([list(word) for word in negative_test_set.sequence.values])
positive_seq_test = np.array([list(word) for word in positive_test_set.sequence.values])

# 给训练集和测试集的样本打上标签
negative_lab1 = np.zeros((negative_seq1.shape[0],), dtype=int)
negative_lab = np.zeros((negative_seq.shape[0],), dtype=int)
positive_lab = np.ones((positive_seq.shape[0],), dtype=int)
negative_labnew = np.zeros((negative_seqnew.shape[0],), dtype=int)
negative_lab_val = np.zeros((negative_seq_val.shape[0],), dtype=int)
positive_lab_val = np.ones((positive_seq_val.shape[0],), dtype=int)
negative_lab_test = np.zeros((negative_seq_test.shape[0],), dtype=int)
positive_lab_test = np.ones((positive_seq_test.shape[0],), dtype=int)

#  平衡数据集

balanced_data = False
token = True
window = 19  # 3 to 19 odd number

if balanced_data:
    df_train_pos = positive_seq
    df_train_neg = negative_seqnew
    df_lab_pos = positive_lab
    df_lab_neg = negative_labnew
else:
    df_train_pos = positive_seq
    df_train_neg = negative_seq
    df_lab_pos = positive_lab
    df_lab_neg = negative_lab

start_w = 9 - int(window / 2)
end_w = 9 + int(window / 2) + 1

# 将正负样本连接到一起组合成正常的训练集和标签
dataset_X = np.concatenate((df_train_pos, df_train_neg), axis=0, out=None)
dataset_Y = np.concatenate((df_lab_pos, df_lab_neg), axis=0, out=None)
dataset_X_val = np.concatenate((positive_seq_val, negative_seq_val), axis=0, out=None)
dataset_Y_val = np.concatenate((positive_lab_val, negative_lab_val), axis=0, out=None)
dataset_X_test = np.concatenate((positive_seq_test, negative_seq_test), axis=0, out=None)
dataset_Y_test = np.concatenate((positive_lab_test, negative_lab_test), axis=0, out=None)

# 平衡验证集
negative_seq_val_eq = negative_seq_val[1:len(positive_seq_val) + 1]  # 为什么不要第一个数据？
negative_lab_val_eq = negative_lab_val[1:len(positive_lab_val) + 1]
dataset_X_val_eq = np.concatenate((positive_seq_val, negative_seq_val_eq), axis=0, out=None)
dataset_Y_val_eq = np.concatenate((positive_lab_val, negative_lab_val_eq), axis=0, out=None)

#  获得初步的词向量矩阵，相当于embedding过程，将所有的序列转化成一个n*19维度的矩阵
asam = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(asam)

#  训练集
dataset_X_token = []
for i in range(len(dataset_X)):
    temp = tokenizer.texts_to_sequences(dataset_X[i])  # 将dataset_X中的字母转化为序列
    dataset_X_token = np.append(dataset_X_token, temp)

dataset_X_token = dataset_X_token - 1
dataset_X_token = dataset_X_token.reshape(len(dataset_X), 19)
dataset_X_token = dataset_X_token[:, range(start_w, end_w)]

# 测试集
dataset_X_token_val = []
for i in range(len(dataset_X_val)):
    temp = tokenizer.texts_to_sequences(dataset_X_val[i])
    dataset_X_token_val = np.append(dataset_X_token_val, temp)

dataset_X_token_val = dataset_X_token_val - 1
dataset_X_token_val = dataset_X_token_val.reshape(len(dataset_X_val), 19)
dataset_X_token_val = dataset_X_token_val[:, range(start_w, end_w)]

# 验证集
dataset_X_token_val_eq = []
for i in range(len(dataset_X_val_eq)):
    temp = tokenizer.texts_to_sequences(dataset_X_val_eq[i])
    dataset_X_token_val_eq = np.append(dataset_X_token_val_eq, temp)

dataset_X_token_val_eq = dataset_X_token_val_eq - 1
dataset_X_token_val_eq = dataset_X_token_val_eq.reshape(len(dataset_X_val_eq), 19)
dataset_X_token_val_eq = dataset_X_token_val_eq[:, range(start_w, end_w)]

# Tokenizing, Unique character got its own number - testing
dataset_X_token_test = []
for i in range(len(dataset_X_test)):
    temp = tokenizer.texts_to_sequences(dataset_X_test[i])
    dataset_X_token_test = np.append(dataset_X_token_test, temp)

dataset_X_token_test = dataset_X_token_test - 1
dataset_X_token_test = dataset_X_token_test.reshape(len(dataset_X_test), 19)
dataset_X_token_test = dataset_X_token_test[:, range(start_w, end_w)]

#  将数据随机打乱
X_train, y_train = shuffle(dataset_X_token, dataset_Y, random_state=13)
X_val, y_val = shuffle(dataset_X_token_val, dataset_Y_val, random_state=13)
X_val_eq, y_val_eq = shuffle(dataset_X_token_val_eq, dataset_Y_val_eq, random_state=13)
X_test, y_test = shuffle(dataset_X_token_test, dataset_Y_test, random_state=13)

# 将数据传输到gpu上面进行运算
X_train_torch = torch.from_numpy(X_train.astype(np.int64)).cuda()
X_val_torch = torch.from_numpy(X_val.astype(np.int64)).cuda()
X_val_eq_torch = torch.from_numpy(X_val_eq.astype(np.int64)).cuda()
X_test_torch = torch.from_numpy(X_test.astype(np.int64)).cuda()

y_train_torch = torch.from_numpy(y_train.astype(np.int64)).cuda()
y_val_torch = torch.from_numpy(y_val.astype(np.int64)).cuda()
y_val_eq_torch = torch.from_numpy(y_val_eq.astype(np.int64)).cuda()
y_test_torch = torch.from_numpy(y_test.astype(np.int64)).cuda()

# 超参数

num_epochs = 1
learning_rate = 0.00001
weight_loss = torch.tensor([0.13, 0.87]).cuda()
criterion = nn.CrossEntropyLoss()  # weight=weight_loss
encoder_number = 2

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x
        # n_heads x len_k x d_v] 首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64)

        # 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        # 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        # 上面代码获取之后得到的pe:[max_len*d_model]

        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(512, 64 * 8)
        self.W_K = nn.Linear(512, 64 * 8)
        self.W_V = nn.Linear(512, 64 * 8)
        self.linear = nn.Linear(8 * 64, 512)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, Q, K, V, attn_mask):
        # 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value; 输入进来的数据形状： Q: [batch_size x len_q x d_model],
        # K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # 下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk， 分成8头，一共64维
        q_s = self.W_Q(Q).view(batch_size, -1, 8, 64).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, 8, 64).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, 8, 64).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x
        # len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, 8, 1, 1)

        # 然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        # 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            8 * 64)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model]
        # 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn_batch = CNNModel()  # 定义CNN块
        self.src_emb = nn.Embedding(21, 512)  # 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(512)  # 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(encoder_number)])  # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder
        # 并没有使用词向量和位置编码，所以抽离出来；
        self.LinearL1 = nn.Linear(512, 2)

    def forward(self, enc_inputs):
        # 这里我们的 enc_inputs 形状是： [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.cnn_batch(enc_outputs)  # 首先经过一个CNN块

        # 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]

        # 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_outputs = enc_outputs.permute(1, 0, 2)
        enc_outputs = torch.sigmoid(self.LinearL1(enc_outputs[enc_outputs.shape[0] - 1]))

        return enc_outputs, enc_self_attns


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.active1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.active2 = nn.ReLU()

        self.drop1 = nn.Dropout(1)
        self.drop2 = nn.Dropout(0.1)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        output = input.view(input.shape[0], 1, input.shape[1], input.shape[2])  # 将数据转化为四维张量[batch_size,
        # channels, height, width]

        output = self.conv1(output)
        output = self.active1(output)
        output = self.drop1(output)

        output = self.conv2(output)
        output = self.active2(output)

        output = torch.squeeze(output, dim=1)
        # print(output)
        # print("outputshape", output.shape)
        return output


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # 编码层
        self.projection = nn.Linear(64, 21,
                                    bias=False)  # 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax

    def forward(self, enc_inputs):
        # # 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size,
        # tgt_len]，主要是作为解码端的输入

        # enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        # enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        return enc_outputs


model = Transformer().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for X_train_torch, y_train_torch in train_data_loader:
        # Forward pass and loss
        y_pred = model(X_train_torch)
        loss = criterion(y_pred, y_train_torch)
        # loss_fn = FocalLoss(gamma=5)
        # loss = loss_fn(y_pred, y_train_torch)
        # loss_fn = WeightedCrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]))
        # loss = loss_fn(y_pred, y_train_torch)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

    # save best epoch
    if loss.item() < np.inf:
        valid_best_loss = loss.item()
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")

print(f'Best Epoch: {best_epoch}')

# validation
model.load_state_dict(torch.load("best_model.pth"))
with torch.no_grad():
    y_predicted_imbalanced = model(X_val_torch)
    y_predicted_balanced = model(X_val_eq_torch)
    acc1 = torch.max(y_predicted_imbalanced, 1)[1].eq(y_val_torch).sum() / float(y_val_torch.shape[0])
    acc2 = torch.max(y_predicted_balanced, 1)[1].eq(y_val_eq_torch).sum() / float(y_val_eq_torch.shape[0])

    f1_score_imbalance = f1_score(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), y_val_torch.cpu().numpy(),
                                  average='macro')
    f1_score_balance = f1_score(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy(),
                                average='macro')

    mcc_imbalance = matthews_corrcoef(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(),
                                      y_val_torch.cpu().numpy())  # y_val_torch 为0
    mcc_balance = matthews_corrcoef(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy())

    fpr, tpr, thresholds = metrics.roc_curve(y_val_torch.cpu().numpy(),
                                             torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), pos_label=1)
    auc_imbalance = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(y_val_eq_torch.cpu().numpy(),
                                             torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), pos_label=1)
    auc_balance = metrics.auc(fpr, tpr)

print(
    f'Validation accuracy (imbalance): {acc1:.4f}, F1: {f1_score_imbalance:.4f}, mcc: {mcc_imbalance:.4f}, auc: {auc_imbalance:.4f}')
print(confusion_matrix(torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), y_val_torch.cpu().numpy()))
print()
print(
    f'Validation accuracy (balance): {acc2:.4f}, F1: {f1_score_balance:.4f}, mcc: {mcc_balance:.4f}, auc: {auc_imbalance:.4f}')
print(confusion_matrix(torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), y_val_eq_torch.cpu().numpy()))

model.load_state_dict(torch.load("best_model.pth"))
with torch.no_grad():
    start = time.time()
    y_predicted_test = model(X_test_torch)
    end = time.time()
    print("Model Time:", end - start)
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
    fpr, tpr, thresholds = metrics.roc_curve(y_test_torch.cpu().numpy(),
                                             torch.max(y_predicted_test, 1)[1].cpu().numpy(), pos_label=1)
    # 通过特异度和灵敏度作为参数得出AUC指标（ROC曲线下面积）
    auc_test = metrics.auc(fpr, tpr)

print(
    f'Test accuracy: {acc_test:.4f}, F1: {f1_score_test:.4f}, mcc: {mcc_test:.4f}, auc: {auc_test:.4f}')
print(confusion_matrix(torch.max(y_predicted_test, 1)[1].cpu().numpy(), y_test_torch.cpu().numpy()))
