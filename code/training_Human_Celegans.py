import pandas as pd
import sys
import torch.nn as nn
from model import Diff_DTA_GIN,Diff_DTA_GCN,Diff_DTA_GAT,Diff_DTA_SAGE
from utils import *
from torch_geometric.data import DenseDataLoader
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from MMDLoss import *
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, a, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        criterion = MMDLoss()
        cl_loss = criterion(x_local, x_global)
        loss_all = loss + 0.05 * l_loss + 0.05 * e_loss + 0.05 * cl_loss
        loss_all.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss_all: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(), loss_all.item()))


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,_,_,_,_,_ = model(data)
            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)

    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()




datasets = [['Human','Celegans'][int(sys.argv[1])]]
model_select = [Diff_DTA_GIN,Diff_DTA_GCN,Diff_DTA_GAT,Diff_DTA_SAGE][int(sys.argv[2])] # 选择模型
model_name = model_select.__name__
print(model_name)
cuda_name = "cuda:0"

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 2048
LR = 0.0005
LOG_INTERVAL = 10
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
ret_list = []
# Main program: iterate over different datasets
for seed in [100,1000,2000]:# 设置随机种子
    same_seeds(seed)
    for dataset in datasets:
        print('\nrunning on ', dataset)
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_val = 'data/processed/' + dataset + '_val.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data_Human_Celegans.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset + '_train')
            val_data = TestbedDataset(root='data', dataset=dataset + '_val')
            test_data = TestbedDataset(root='data', dataset=dataset + '_test')

            # make data PyTorch mini-batch processing ready
            train_loader = DenseDataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DenseDataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            test_loader = DenseDataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = model_select()
            model = model.to(device)
            # model.load_state_dict(torch.load('model_GINConvNet_davis.model')) # 导入网络的参数
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_roc = 0
            best_epoch = -1
            model_file_name = 'model_' + dataset + '_' + model_name + '.model'
            for epoch in range(NUM_EPOCHS):
                time_begin = time.time()
                train(model, device, train_loader, optimizer, epoch + 1)
                G, P, pred = predicting(model, device, val_loader)
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr), precision_score(G,pred), recall_score(G,pred)]
                test_roc = ret[0]
                test_prc = ret[1]
                if ret[0] > best_roc:
                    torch.save(model.state_dict(), model_file_name)
                    best_epoch = epoch + 1
                    best_roc = ret[0]
                    G, P, pred = predicting(model, device, test_loader)
                    G_ = G
                    P_ = P
                    best_ret = [roc_auc_score(G, P), auc(fpr, tpr), precision_score(G,pred), recall_score(G,pred)]
                    print('rmse improved at epoch ', best_epoch, '; best_AUROC, best_AUPRC, best_precision, best_recall:', best_ret[0]
                          , best_ret[1],best_ret[2],best_ret[3], dataset)
                else:
                    print(ret[0], 'No improvement since epoch ',  best_epoch, '; best_AUROC, best_AUPRC, best_precision, best_recall:', best_ret[0]
                          , best_ret[1],best_ret[2],best_ret[3], dataset)
                time_end = time.time()
                print("spend time：", time_end - time_begin, "s")
    ret_list.append(best_ret)
    d = pd.DataFrame(ret_list)
    d.to_csv(dataset+'_'+model_name+'_random.csv',index=0)
