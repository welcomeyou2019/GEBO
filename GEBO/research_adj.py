import numpy as np
from LDS_GNN_modify_local.lds_gnn.analysis_adj import load_data, reorganize_data_for_es, normalize_adj
import matplotlib.pyplot as plt

def norm(x, dim=0):
    return x / np.sum(x, axis=dim).reshape(-1,1)

def plot(adj, j, noise):
    plt.plot(range(len(adj)),adj)
    if len(noise) == 0:
        plt.title(str(j))
    else:
        plt.title(str(j)+'_'+str(noise))
    plt.show()

dataset_name = 'cora'
res = load_data(dataset_str=dataset_name)
adj, y_train, y_val, y_val_sep, y_es, y_test, features, train_mask, val_mask, es_mask, test_mask, \
    nclass, idx_train, idx_val, idx_test, labels= reorganize_data_for_es(res, seed=1979, dataset_name=dataset_name)

labels = np.argmax(labels, 1)
train_weight = np.load('./parameter/Z_gsage_'+dataset_name+'_5_0.npy')
train_weight = np.load('./parameter/Z_'+dataset_name+'25_constrain_4.npy')
adj_weight = normalize_adj(adj).todense()

# adj = adj.todense()
# adj  =adj -np.eye(adj.shape[0])
# # print(adj)
# row, col = np.where(adj!=0)
# inter = 0
# intra = 0
# for i,j in zip(row, col):
#     if (labels[i] != labels[j]).any():
#         inter+=1
#     else:
#         intra +=1
# print(inter,intra, inter+intra, intra/(inter+intra))  #inter ratio   cora: 0.151   citeseer:0.194    terrorist:0.362/0.435   airport:0.289   ppi:

delta = train_weight - adj_weight
row, col = np.where(delta < 0) #权重降低


# test_acc, out,_,_ = train_GCN(dataset_name, train_weight)
# print(test_acc)
# print(out.shape)
# pred = np.argmax(out, 1)

soft_train_weight = norm(train_weight, dim=1)
print(soft_train_weight)
for j in range(1708,2708):
    noise = []
    lie = np.where(soft_train_weight[j,:] != 0)[0]
    for i in lie:
        if labels[j] != labels[i]:
            noise.append(i)
    # plot(soft_train_weight[j,:],j,noise)
    if pred[j] == labels[j]:
        # print(j, noise, len(lie), pred[j] == labels[j])
        continue
    else:
        print(j, noise, len(lie))
        plot(soft_train_weight[j,:],j,noise)
# acc,_,_,_ = train_GCN(dataset_name, train_weight)
# print(acc)