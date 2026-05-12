import pandas as pd
import sys
import torch.nn as nn
from model import Diff_DTA_GIN,Diff_DTA_GCN,Diff_DTA_GAT,Diff_DTA_SAGE
from sklearn.metrics import r2_score
from utils import *
from torch_geometric.data import DenseDataLoader
import time
from MMDLoss import *
# fix random seed
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
    loss_train = 0
    num_sample = 0
    a_all = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, a, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        criterion = MMDLoss()
        cl_loss = criterion(x_local, x_global)
        loss_all = loss + 0.05 * l_loss + 0.05 * e_loss + 0.05 * cl_loss
        loss_all.backward()
        loss_train = loss_train + data.y.shape[0] * loss.item()
        num_sample = num_sample + data.y.shape[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss_all: {:.6f}'.format(epoch,
            batch_idx * len(data.x),
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item(), loss_all.item()))
        a = a.cpu().detach().numpy().sum(axis=0)/data.y.shape[0]
        a_all.append(list(a.flatten()))
    print("注意力分数 x_H_1, x_H_2, xt, x_G:", np.array(a_all).sum(axis=0)/len(a_all))
    print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, loss_train / num_sample))
    return loss_train / num_sample, np.array(a_all).sum(axis=0)/len(a_all)
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # output = model(data)
            output,_,_,_,_,_ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels,total_preds


datasets = [['davis','kiba'][int(sys.argv[1])]] 
model_select = [Diff_DTA_GIN,Diff_DTA_GCN,Diff_DTA_GAT,Diff_DTA_SAGE][int(sys.argv[2])] # 选择模型
model_name = model_select.__name__
print(model_name)
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
mse_list = []
ci_list = []
r2_list = []
a_list = []
for seed in [100,1000,2000]:# 设置随机种子
    same_seeds(seed)
    # Main program: iterate over different datasets
    loss_train_list = []
    loss_test_list = []
    for dataset in datasets:
        print('\nrunning on ', dataset )
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data_davis_kiba.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test')

            # make data PyTorch mini-batch processing ready
            train_loader = DenseDataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            test_loader = DenseDataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            if dataset=='kiba':
                train_data1 = TestbedDataset(root='data', dataset=dataset+'_train1')
                test_data1 = TestbedDataset(root='data', dataset=dataset+'_test1')
                # make data PyTorch mini-batch processing ready
                train_loader1 = DenseDataLoader(train_data1, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
                test_loader1 = DenseDataLoader(test_data1, batch_size=TEST_BATCH_SIZE, shuffle=False)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            # n1 n2
            default_nodes = {
                'davis': (4, 2),
                'kiba': (6, 2),
            }
            n1_default, n2_default = default_nodes[dataset]
            model = model_select(num_nodes_1=n1_default, num_nodes_2=n2_default)
            model = model.to(device)
            # model.load_state_dict(torch.load('model_GINConvNet_davis.model')) # 导入网络的参数
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000000
            best_ci = 0
            best_epoch = -1
            model_file_name = 'model_' + dataset + '_' + model_name + '_' + str(seed) + '.model'
            for epoch in range(NUM_EPOCHS):
                time_begin = time.time()
                if dataset=='kiba':
                    loss_train,a  = train(model, device, train_loader, optimizer, epoch+1)
                    loss_train1,a1 = train(model, device, train_loader1, optimizer, epoch+1)
                    loss_train = (loss_train * 19576 + loss_train1 * 133) / (19576+133)
                    loss_train_list.append(loss_train)
                    a = (a * 19576 + a1 * 133) / (19576+133)
                    a_list.append(list(a))
                else:
                    loss_train,a = train(model, device, train_loader, optimizer, epoch+1)
                    loss_train_list.append(loss_train)
                    a_list.append(list(a))
                if dataset=='kiba':
                    G,P = predicting(model, device, test_loader)
                    G1,P1 = predicting(model, device, test_loader1)
                    G = torch.cat((G, G1), 0)
                    P = torch.cat((P, P1), 0)
                else:
                    G,P = predicting(model, device, test_loader)
                G = G.numpy().flatten()
                P = P.numpy().flatten()
                ret = [mse(G,P), get_rm2(G,P)]
                loss_test_list.append(ret[0])
                if ret[0]<best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    best_epoch = epoch+1
                    best_mse = ret[0]
                    best_r2 = ret[1]
                    G_ = G
                    P_ = P
                    print('rmse improved at epoch ', best_epoch, '; best_mse, rm2:', best_mse, best_r2 ,dataset)
                else:
                    print(ret[0],'No improvement since epoch ', best_epoch, '; best_mse, rm2:', best_mse, best_r2, dataset)
                time_end = time.time()
                print("spend time：", time_end - time_begin, "s")
                d = pd.DataFrame(loss_train_list,columns=['train_loss'])
                d['test_loss'] = loss_test_list
                d.to_csv(dataset + "训练损失_" + str(seed) + "_" + model_name + ".csv",index=0)
                d = pd.DataFrame(a_list,columns=['x_H_1', 'x_H_2', 'xt', 'x_G'])
                d.to_csv(dataset + "注意力分数_" + str(seed) + "_" + model_name + ".csv",index=0)
    mse_list.append(best_mse)
    ci_list.append(ci(G_,P_))
    r2_list.append(get_rm2(G_,P_))
    d = pd.DataFrame(mse_list,columns=['mse'])
    d['ci'] = ci_list
    d['r2'] = r2_list
    d.to_csv(dataset+'_'+model_name+'_random.csv',index=0)
