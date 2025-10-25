# python3.7 pytorch_GPU 1.13.1+cu117
# -*- coding: UTF-8 -*-
# @IDE    :
# @Author : keymon
# @Date   : 2023-10-31 22:05

# import os
import time
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
# from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.backends.cudnn.benchmark = False

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
# negative_seq = np.concatenate((negative_seq1,
#                                negative_seq2,
#                                negative_seq3,
#                                negative_seq4,
#                                negative_seq5), axis=0, out=None)
negative_seq = negative_seq1
negative_seqnew = np.concatenate((negative_seq1,
                                  negative_seq2), axis=0, out=None)
positive_seq1 = np.array([list(word) for word in positive_training_set1.sequence.values])
positive_seq2 = np.array([list(word) for word in positive_training_set2.sequence.values])
# positive_seq = np.concatenate((positive_seq1,
#                                positive_seq2,), axis=0, out=None)
positive_seq = positive_seq1

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
X_train_torch = torch.from_numpy(X_train.astype(np.int64))
X_val_torch = torch.from_numpy(X_val.astype(np.int64))
X_val_eq_torch = torch.from_numpy(X_val_eq.astype(np.int64))
X_test_torch = torch.from_numpy(X_test.astype(np.int64))

y_train_torch = torch.from_numpy(y_train.astype(np.int64))
y_val_torch = torch.from_numpy(y_val.astype(np.int64))
y_val_eq_torch = torch.from_numpy(y_val_eq.astype(np.int64))
y_test_torch = torch.from_numpy(y_test.astype(np.int64))

# 超参数
num_epochs = 100
learning_rate = 0.00001
criterion = nn.CrossEntropyLoss()

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model_which = 'modelLSTM'  # 'model1' 'modelCNN' 'modelLSTM'


# Create the model of lstm

class ModelLSTM(nn.Module):
    def __init__(self):
        super(ModelLSTM, self).__init__()
        self.embedding = nn.Embedding(len(asam), 21)
        self.lstm1 = nn.LSTM(21, 64, 1)
        self.lstm2 = nn.LSTM(64, 128, 1)
        self.lstm3 = nn.LSTM(128, 256, 1)
        self.lstm4 = nn.LSTM(256, 512, 1)
        self.dropL1 = nn.Dropout(p=0.5)
        self.dropL2 = nn.Dropout(p=0.5)
        self.dropL3 = nn.Dropout(p=0.5)
        self.dropL4 = nn.Dropout(p=0.5)
        self.linearL1 = nn.Linear(512, 512)
        self.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.embedding(x)
        lstm = out.permute(1, 0, 2)
        # lstm = out.permute(19, 1024, 19)
        lstm = self.lstm1(lstm)
        lstm = self.dropL1(lstm[0])
        lstm = self.lstm2(lstm)
        lstm = self.dropL2(lstm[0])
        lstm = self.lstm3(lstm)
        lstm = self.dropL3(lstm[0])
        lstm = self.lstm4(lstm)
        lstm = self.dropL4(lstm[0])
        lstm = self.linearL1(lstm[lstm.shape[0] - 1])
        out = self.fc(lstm)
        out = self.softmax(out)
        return out


# define the optim
model = ModelLSTM()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
inputType = np.int64
output_onehot = False
output_type = np.int64

best_loss = np.inf
best_epoch = 0
for epoch in range(num_epochs):
    print(f'epoch: {epoch}')
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
    # if (loss.item() < valid_best_loss):
    #     valid_best_loss = loss.item()
    #     best_epoch = epoch
    #     torch.save(model.state_dict(), "best_model.pth")
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
print(f'Best Epoch: {best_epoch}')
model.load_state_dict(torch.load("best_model.pth"))
with torch.no_grad():
    start = time.time()
    y_predicted_imbalanced = model(X_val_torch)
    end = time.time()
    print("ModelTime:", end-start)
    y_predicted_balanced = model(X_val_eq_torch)
    acc1 = torch.max(y_predicted_imbalanced, 1)[1].eq(y_val_torch).sum() / float(y_val_torch.shape[0])
    acc2 = torch.max(y_predicted_balanced, 1)[1].eq(y_val_eq_torch).sum() / float(y_val_eq_torch.shape[0])

    f1_score_imbalance = f1_score(torch.max(y_predicted_imbalanced, 1)[1].numpy(), y_val_torch.numpy(),
                                  average='macro')
    f1_score_balance = f1_score(torch.max(y_predicted_balanced, 1)[1].numpy(), y_val_eq_torch.numpy(),
                                average='macro')

    mcc_imbalance = matthews_corrcoef(torch.max(y_predicted_imbalanced, 1)[1].numpy(),
                                      y_val_torch.numpy())  # y_val_torch 为0
    mcc_balance = matthews_corrcoef(torch.max(y_predicted_balanced, 1)[1].numpy(), y_val_eq_torch.numpy())

    # fpr, tpr, thresholds = metrics.roc_curve(y_val_torch.cpu().numpy(),
    #                                          torch.max(y_predicted_imbalanced, 1)[1].cpu().numpy(), pos_label=1)
    # auc_imbalance = metrics.auc(fpr, tpr)
    # fpr, tpr, thresholds = metrics.roc_curve(y_val_eq_torch.cpu().numpy(),
    #                                          torch.max(y_predicted_balanced, 1)[1].cpu().numpy(), pos_label=1)
    # auc_balance = metrics.auc(fpr, tpr)
    auroc_imbalance = roc_auc_score(torch.max(y_predicted_imbalanced, 1)[1].numpy(), y_val_torch.numpy())
    auroc_balance = roc_auc_score(torch.max(y_predicted_balanced, 1)[1].numpy(), y_val_eq_torch.numpy())

    auprc_imbalance = average_precision_score(torch.max(y_predicted_imbalanced, 1)[1].numpy(), y_val_torch.numpy())
    auprc_balance = average_precision_score(torch.max(y_predicted_balanced, 1)[1].numpy(), y_val_eq_torch.numpy())

# print(f'Validation accuracy (imbalance): {acc1:.4f}, F1: {f1_score_imbalance:.4f}, mcc: {mcc_imbalance:.4f}, auc: {auc_imbalance:.4f}')

print(f'Validation accuracy (imbalance): {acc1:.4f}, F1: {f1_score_imbalance:.4f}, mcc: {mcc_imbalance:.4f}, auroc: {auroc_imbalance:.4f}, auprc: {auprc_imbalance:.4f}')
print(confusion_matrix(torch.max(y_predicted_imbalanced, 1)[1].numpy(), y_val_torch.numpy()))
print()
# print(f'Validation accuracy (balance): {acc2:.4f}, F1: {f1_score_balance:.4f}, mcc: {mcc_balance:.4f}, auc: {auc_imbalance:.4f}')
print(f'Validation accuracy (balance): {acc2:.4f}, F1: {f1_score_balance:.4f}, mcc: {mcc_balance:.4f}, auroc: {auroc_balance:.4f}, auprc: {auprc_balance:.4f}')
print(confusion_matrix(torch.max(y_predicted_balanced, 1)[1].numpy(), y_val_eq_torch.numpy()))
print('modelLSTM')
modelLSTM_pred_val_imbalance = nn.Softmax()(y_predicted_imbalanced)
modelLSTM_pred_val_balance = nn.Softmax()(y_predicted_balanced)

modelCNN_pred_val_imbalance = []
modelCNN_pred_val_balance = []

if model_which == 'modelCNN':
    print('modelCNN')
    modelCNN_pred_val_imbalance = nn.Softmax()(y_predicted_imbalanced)
    modelCNN_pred_val_balance = nn.Softmax()(y_predicted_balanced)


elif model_which == 'modelLSTM':
    print('modelLSTM')
    modelLSTM_pred_val_imbalance = nn.Softmax()(y_predicted_imbalanced)
    modelLSTM_pred_val_balance = nn.Softmax()(y_predicted_balanced)
else:
    print('not running softmax')

if (modelCNN_pred_val_imbalance != []) & (modelLSTM_pred_val_imbalance != []):
    print('both executed')
    ensamble_pred_imbalance = (modelLSTM_pred_val_imbalance * 0.17) + (modelCNN_pred_val_imbalance * 0.83)
    ensamble_pred_balance = (modelLSTM_pred_val_balance * 0.17) + (modelCNN_pred_val_balance * 0.83)

    acc1 = torch.max(ensamble_pred_imbalance, 1)[1].eq(y_val_torch).sum() / float(y_val_torch.shape[0])
    acc2 = torch.max(ensamble_pred_balance, 1)[1].eq(y_val_eq_torch).sum() / float(y_val_eq_torch.shape[0])
    f1_score_imbalance = f1_score(torch.max(ensamble_pred_imbalance, 1)[1].numpy(), y_val_torch.numpy(),
                                  average='macro')
    f1_score_balance = f1_score(torch.max(ensamble_pred_balance, 1)[1].numpy(), y_val_eq_torch.numpy(),
                                average='macro')

    print()
    print(f'Validation accuracy (imbalance): {acc1.item():.4f}, F1: {f1_score_imbalance.item():.4f}')
    print(confusion_matrix(torch.max(ensamble_pred_imbalance, 1)[1].numpy(), y_val_torch.numpy()))
    print(f'Validation accuracy (balance): {acc2.item():.4f}, F1: {f1_score_balance.item():.4f}')
    print(confusion_matrix(torch.max(ensamble_pred_balance, 1)[1].numpy(), y_val_eq_torch.numpy()))

else:
    print('one of the model or both, have not been executed yet')

model.load_state_dict(torch.load("best_model.pth"))
with torch.no_grad():
    y_predicted_test = model(X_test_torch)
    # 准确率（预测准确的正样本+预测准确的负样本/总样本）
    acc_test = torch.max(y_predicted_test, 1)[1].eq(y_test_torch).sum() / float(y_test_torch.shape[0])
    # F1（2*准确率*召回率/准确率+召回率）召回率=灵敏度
    # 模型精确率和召回率的一种调和平均，它的最大值是1，最小值是0
    f1_score_test = f1_score(torch.max(y_predicted_test, 1)[1].numpy(), y_test_torch.numpy(),
                             average='macro')
    # MCC它的取值范围为[-1,1]，取值为1时表示对受试对象的完美预测，取值为0时表示预测的结果还不如随机预测的结果，-1是指预测分类和实际分类完全不一致
    mcc_test = matthews_corrcoef(torch.max(y_predicted_test, 1)[1].numpy(), y_test_torch.numpy())
    # 绘制ROC曲线，获得特异度和灵敏度（假正率和真正率）和用于计算特异度和灵敏度的阈值
    # 特异度（预测准确的正样本/正样本） 灵敏度（预测准确的负样本/负样本）
    # fpr, tpr, thresholds = metrics.roc_curve(y_test_torch.cpu().numpy(),
    #                                          torch.max(y_predicted_test, 1)[1].cpu().numpy(), pos_label=1)
    # # 通过特异度和灵敏度作为参数得出AUC指标（ROC曲线下面积）
    # auc_test = metrics.auc(fpr, tpr)
    auroc_test = roc_auc_score(torch.max(y_predicted_test, 1)[1].numpy(), y_test_torch.numpy())
    auprc_test = average_precision_score(torch.max(y_predicted_test, 1)[1].numpy(), y_test_torch.numpy())

# print(f'Test accuracy: {acc_test:.4f}, F1: {f1_score_test:.4f}, mcc: {mcc_test:.4f}, auc: {auc_test:.4f}')
print(f'Test accuracy: {acc_test:.4f}, F1: {f1_score_test:.4f}, mcc: {mcc_test:.4f}, auroc: {auroc_test:.4f}, auprc: {auprc_test:.4f}')
print(confusion_matrix(torch.max(y_predicted_test, 1)[1].numpy(), y_test_torch.numpy()))
