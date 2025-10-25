import pandas as pd
import numpy as np
# 磷酸化数据集
negative_training_set_txt = pd.read_csv('dataset/ST_data/ST_negative_training_set_19(4308).txt', sep="\n", header=None)
positive_training_set_txt = pd.read_csv('dataset/ST_data/ST_positive_training_set_19(4308).txt', sep="\n", header=None)
negative_test_set_txt = pd.read_csv('dataset/ST_data/ST_negative_test_set_19(1079).txt', sep="\n", header=None)
positive_test_set_txt = pd.read_csv('dataset/ST_data/ST_positive_test_set_19(1079).txt', sep="\n", header=None)

negative_seq = np.array([list(word) for word in negative_training_set_txt])
positive_seq = np.array([list(word) for word in positive_training_set_txt])
negative_seq_test = np.array([list(word) for word in negative_test_set_txt])
positive_seq_test = np.array([list(word) for word in positive_test_set_txt])

negative_lab = np.zeros((negative_seq.shape[0],), dtype=int)
positive_lab = np.ones((positive_seq.shape[0],), dtype=int)
negative_lab_test = np.zeros((negative_seq_test.shape[0],),dtype=int)
positive_lab_test = np.ones((positive_seq_test.shape[0],),dtype=int)


