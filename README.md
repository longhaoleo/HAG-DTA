# HAG-DTA

drug-target binding affinity prediction with hierarchical multi-scale attention fusion graph neural networks

## Step1：配置

对”服务器配置“文件夹中的文件进行配置，运行以下命令：

```bash
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
```

pytorch 2.0.0版本，python3.8版本

## Step2：数据处理

运行以下程序：

```python
python create_data_davis_kiba.py
python create_data_Human_Celegans.py
```

构建数据集

## Step3：进行训练

程序training_davis_kiba.py是训练davis和kiba数据集的

程序training_Human_Celegans.py是训练Human和Celegans数据集的

训练示例：

```python
python training_davis_kiba.py 0 1
python training_davis_kiba.py dataset_id model_id
```

其中：

```
dataset_id: 0=davis, 1=kiba
model_id: 0=GIN, 1=GCN, 2=GAT, 3=SAGE
```