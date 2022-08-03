import pandas as pd
import numpy as np
# 导入数据：分隔符为空格
raw_data = pd.read_csv('./terrorist_attack.nodes',sep = '\t',header = None)
num = raw_data.shape[0]

a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b,a)
map = dict(c)

features =raw_data.iloc[:,1:-1]

print(features.shape)

labels = pd.get_dummies(raw_data[107])
labels = np.array(labels)
a = np.argmax(labels, 1)
lab_dic = {i:[] for i in range(6)}
for ind,i in enumerate(a):
    lab_dic[i].append(ind)

idx_train = []
idx_val = []
idx_test = []
for i in lab_dic:
    idx_train.extend(lab_dic[i][:int(0.1*len(lab_dic[i]))])
    idx_val.extend(lab_dic[i][int(0.1*len(lab_dic[i])):int(0.3*len(lab_dic[i]))])
    idx_test.extend(lab_dic[i][int(0.3*len(lab_dic[i])):])
print(len(idx_train))

raw_data_cites = pd.read_csv('terrorist_attack_loc.edges',sep = '\\s+',header = None)
raw_data_cites_1 = pd.read_csv('terrorist_attack_loc_org.edges', sep='\\s+', header=None)
adj = np.zeros((num,num))
# 创建邻接矩阵
for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
    x = map[i]
    y = map[j]  #替换论文编号为[0,2707]
    adj[x][y] = adj[y][x] = 1 #有引用关系的样本点之间取1
# 查看邻接矩阵的元素和（按每列汇总）
for i, j in zip(raw_data_cites_1[0], raw_data_cites_1[1]):
    x = map[i]
    y = map[j]  # 替换论文编号为[0,2707]
    adj[x][y] = adj[y][x] = 1  # 有引用关系的样本点之间取1
print(np.sum(adj))