# HAG-DTA 模型阅读指南

这份 README 不是运行手册，而是给接手项目的人看的。

目标只有一个：让你用最短时间弄清楚这个模型是怎么跑起来的，后面方便补实验、改模型、查结果。

## 先说结论：模型应该从哪里看起

不要一上来就从 `model.py` 第一行往下硬读。

更高效的顺序是：

1. 先看训练入口，弄清楚任务是什么、输入输出是什么
2. 再看 `forward()`，弄清楚一次前向到底走了哪几条分支
3. 再回头看数据预处理，弄清楚 `data.x / data.adj / data.target / data.mask` 是怎么来的
4. 最后再看各个子模块的实现细节

对应文件顺序建议如下：

1. `training_davis_kiba.py` 或 `training_Human_Celegans.py`
2. `model.py` 里的 `Diff_DTA_*` 类和 `forward()`
3. `utils.py` 里的 `TestbedDataset`
4. `create_data_davis_kiba.py` / `create_data_Human_Celegans.py`
5. `MMDLoss.py`

如果你现在的目标是“先理解模型结构”，建议先从 `training_davis_kiba.py` 和 `model.py` 开始。

## 仓库里几类文件分别干什么

`code_leo/` 目录下你主要会反复看这几个文件：

- `model.py`
  - 核心模型定义都在这里。
  - 包含 4 个版本：`Diff_DTA_GIN`、`Diff_DTA_GCN`、`Diff_DTA_GAT`、`Diff_DTA_SAGE`。
  - 四个版本整体框架一样，只是图卷积算子不同。

- `training_davis_kiba.py`
  - 回归任务训练脚本。
  - 数据集是 `davis` 和 `kiba`。
  - 指标主要看 `mse`、`ci`、`rm2`。

- `training_Human_Celegans.py`
  - 二分类任务训练脚本。
  - 数据集是 `Human` 和 `Celegans`。
  - 指标主要看 `AUROC`、`AUPRC`、`precision`、`recall`。

- `create_data_davis_kiba.py`
  - 把 DeepDTA 风格原始数据转成 CSV，再转成 PyG 可读的 `.pt`。
  - 药物是图，蛋白是长度 1000 的序列编码。

- `create_data_Human_Celegans.py`
  - Human/Celegans 的数据预处理。
  - 会自己做 train/val/test 划分。

- `utils.py`
  - `TestbedDataset` 在这里，是数据装载的关键。
  - 评估指标函数也在这里。

- `MMDLoss.py`
  - 额外的分布对齐损失。

## 推荐阅读路径

### 第一遍：先建立全局图

先看训练脚本，不要先看子模块实现。

建议按下面顺序：

1. 看 `training_davis_kiba.py` 里选数据集和模型的部分
2. 看 `train()` 里模型返回了什么
3. 看 `predicting()` 里预测时真正用了什么
4. 跳到 `model.py` 里对应的 `Diff_DTA_*` 的 `forward()`

你第一遍需要回答的其实只有 4 个问题：

1. 输入是什么
2. 模型输出是什么
3. 总损失由哪些部分组成
4. 用什么指标挑最好模型

### 第二遍：只读 `forward()`

读 `model.py` 时，先只盯着 `forward()`，先别陷进每个类的细节。

以 `Diff_DTA_GIN` 为例，它的主干可以概括成 4 条分支：

1. 药物局部分支 1
   - 原子图经过第一层 `diff_pool`
   - 得到第一层局部表示 `x_1`

2. 药物局部分支 2
   - 在第一层池化结果上再做一次 `diff_pool`
   - 得到更粗粒度的局部表示 `x_2`

3. 药物全局分支
   - 另一条图分支提取全局图表示 `x_global`

4. 蛋白序列分支
   - `Embedding + TextCNN`
   - 得到序列表示 `xt`

最后把 `x_1`、`x_2`、`xt`、`x_global` 四路表示做注意力融合，再接全连接层输出。

所以这个模型的名字如果用一句话说清楚，就是：

**药物图的多尺度局部表示 + 药物全局表示 + 蛋白序列表示，通过注意力融合做 DTA 预测。**

## 模型细节：建议你这样对着 `forward()` 看

如果你已经打开 `Diff_DTA_GIN.forward()`，建议不要只记“有四个分支”，而是按下面这条主线去看：

1. 输入先被拆成药物图和蛋白序列两部分
2. 药物图同时走“局部层次分支”和“全局分支”
3. 蛋白序列单独走一条 CNN 分支
4. 四路 128 维表示做注意力加权融合
5. 融合结果送进 MLP 输出最终预测

### 1. 输入张量在 `forward()` 里的含义

模型开头读入的是：

```python
x, adj, target, mask = data.x, data.adj, data.target, data.mask
target = target.view(-1, 1000)
```

可以先把它们理解成：

- `x`
  - 药物图的节点特征矩阵
  - 形状大致是 `batch_size × num_nodes × 78`
- `adj`
  - 稠密邻接矩阵
  - 形状大致是 `batch_size × num_nodes × num_nodes`
- `target`
  - 蛋白序列编码
  - 每条序列被截断或补齐到 1000
- `mask`
  - 稠密图里的有效节点掩码
  - 因为 `ToDense(...)` 后会有 padding 节点

这里最关键的一点是：这个项目不是直接在稀疏图上做普通 GNN，而是先把图转成稠密图，再配合 `dense_diff_pool` 做层次池化。

### 2. 药物全局分支具体怎么走

全局分支对应这段逻辑：

```python
s_global = self.global_gnn_pool(x, adj, mask)
x_global = self.global_gnn(x, adj, mask)
x_global = self.attention3(x_global)
x_global, _, _, _ = dense_diff_pool(x_global, adj, s_global, mask)
x_global = x_global.view(-1, 64)
x_global = F.relu(self.lin3(x_global))
```

可以把它拆成 4 步：

1. `global_gnn_pool` 生成 assignment matrix
   - 它输出每个节点应该被分配到哪个簇
   - 对 GIN/GCN/GAT/SAGE 四个版本来说，这里都只输出 1 个簇

2. `global_gnn` 提取节点级全局特征
   - 这条支路会堆多层图卷积
   - 中间不是简单覆盖，而是用了一个门控更新：
     - `mol_z = sigmoid(fc1(x) + fc2(mol_x) + bias)`
     - 再做 `mol_z * x + (1 - mol_z) * mol_x`
   - 这相当于在每一层卷积后，保留一部分旧表示，融合一部分新表示

3. `attention3` 对节点表示做一次自注意力
   - 它不是 transformer 那种多头结构
   - 本质上是对当前节点序列做 `QK^T` 打分，再加权 `V`

4. `dense_diff_pool` 把整张图压缩成 1 个图级簇
   - 因为分配簇数是 1，所以最后 `x_global` 会被压成每个样本一个 64 维向量
   - 再经过 `lin3` 映射成 128 维，方便后面和其他分支对齐

这条分支的作用不是“再做一次局部池化”，而是显式抽一个图级全局语义向量。

### 3. 药物局部层次分支具体怎么走

局部分支是这个模型的主体，核心是两次 `diff_pool`。

第一层：

```python
s = self.gnn1_pool(x, adj, mask)
x_1 = self.gnn1_embed(x, adj, mask)
x_1, adj, l1, e1 = dense_diff_pool(x_1, adj, s, mask)
```

第二层：

```python
s = self.gnn2_pool(x_1, adj)
x_2 = self.gnn2_embed(x_1, adj)
x_2, adj, l2, e2 = dense_diff_pool(x_2, adj, s)
```

这里建议你按“pool 网络”和“embed 网络”分开理解。

- `gnn*_pool`
  - 专门负责生成 assignment matrix
  - 决定节点被聚成几个超节点

- `gnn*_embed`
  - 专门负责生成被聚合的节点表示
  - 决定节点内容是什么

也就是说，每一层 `diff_pool` 都需要两样东西：

1. 节点表示 `Z`
2. 聚类分配矩阵 `S`

然后 `dense_diff_pool(Z, A, S)` 做三件事：

1. 把原图节点聚合成更少的簇节点
2. 同时更新聚合后的邻接矩阵
3. 返回两个正则项损失
   - `l_loss`
   - `e_loss`

对当前默认设置（按论文 per-dataset 最优，可通过环境变量 HAG_DTA_N1/HAG_DTA_N2 覆盖）来说：
- Davis: n1=4, n2=2
- KIBA: n1=6, n2=2
- Human / C.elegans: n1=7, n2=3

所以你可以把两层局部分支理解成：

- `x_1`
  - 第一层粗化后的局部结构表示
- `x_2`
  - 第二层进一步压缩后的高层局部表示

后面作者没有直接拿池化后的结果就输出，而是分别再做：

```python
x_1 = self.attention1(x_1)
x_2 = self.attention2(x_2)
```

再展平：

```python
x_1 = x_1.view(x_1.size(0), -1)
x_2 = x_2.view(x_2.size(0), -1)
```

最后映射到统一的 128 维：

```python
x_1 = F.relu(self.lin1(x_1))
x_2 = F.relu(self.lin2(x_2))
```

所以局部分支真正输出给融合模块的，不是原始的簇节点矩阵，而是两个压平后的 128 维摘要向量。

### 4. 蛋白序列分支具体怎么走

蛋白分支是典型的 `Embedding + TextCNN`：

```python
embedded_xt = self.embedding_xt(target)
embedded_xt = embedded_xt.unsqueeze(1)
embedded_xt = [relu(conv(embedded_xt)).squeeze(3) for conv in self.convs]
embedded_xt = [max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in embedded_xt]
conv_xt = torch.cat(embedded_xt, dim=1)
xt = self.fc1_xt(conv_xt)
```

可以按下面理解：

1. `embedding_xt`
   - 把蛋白序列中每个氨基酸 token 编成 128 维向量

2. `unsqueeze(1)`
   - 把输入变成 `Conv2d` 能接受的格式

3. 三个卷积核 `[3, 4, 5]`
   - 分别提取不同长度的局部 motif
   - 每个卷积核都输出 128 个通道

4. `max_pool1d`
   - 对每个通道做全局最大池化
   - 相当于从每种模式里取最强响应

5. `torch.cat`
   - 三组卷积输出拼起来，得到 `3 × 128 = 384` 维

6. `fc1_xt`
   - 再压到 128 维，和其他三路表示对齐

这条分支没有做复杂序列建模，没有 RNN、Transformer 或 cross-attention，本质上就是一个多尺度 CNN 编码器。

### 5. 四路注意力融合到底怎么做

融合部分最容易被误读成“把四路向量拼接后过线性层”，但作者其实先算了权重：

```python
a = self.attention(x_1, x_2, xt, x_global)
emb = torch.stack([x_1, x_2, xt, x_global], dim=1)
a = a.unsqueeze(dim=2)
xc = (a * emb).reshape(-1, 4 * 128)
```

这里真正发生的是：

1. `Attention_Fusion` 分别对四路表示各算一个分数
2. 四个分数做 `softmax`
3. 得到每个样本的四路权重 `a`
4. 用 `a` 对四路 128 维表示逐路加权
5. 再把加权后的结果拼成 `4 * 128 = 512` 维

注意这里不是“加权求和变成一个 128 维向量”，而是“每一路先乘自己的权重，再保留四路拼接信息”。

这意味着：

- 注意力起的是“重标定四路重要性”的作用
- 但模型并没有丢掉每一路独立的表示槽位

这个设计比简单求和保留了更多结构信息。

### 6. 预测头怎么走

融合后的 `xc` 会经过两层 MLP：

```python
xc = self.fc1(xc)
xc = self.bns1(xc)
xc = self.relu(xc)
xc = F.dropout(xc, p=0.2, training=self.training)

xc = self.fc2(xc)
xc = self.bns2(xc)
xc = self.relu(xc)
xc = F.dropout(xc, p=0.2, training=self.training)

out = self.out(xc)
```

维度关系可以记成：

- `512 -> 1024 -> 256 -> 1`

其中：

- 回归任务里这个 `1` 直接当预测值
- 分类任务里这个 `1` 是 logit，外面再过 `sigmoid`

### 7. `forward()` 返回值为什么这样设计

最后一行是：

```python
return out, l1 + l2, e1 + e2, a, x_2, x_global
```

它不是只为了“多返回点中间结果看看”，而是训练上真的会用：

- `out`
  - 主任务预测
- `l1 + l2`
  - 两层池化的链接重建损失
- `e1 + e2`
  - 两层池化的熵正则
- `a`
  - 用来统计四路注意力分数
- `x_2`
  - 当作局部表示，和 `x_global` 做 MMD
- `x_global`
  - 当作全局表示，和 `x_2` 做 MMD

这里有个实现上的选择值得记住：

- MMD 用的是 `x_2` 和 `x_global`
- 不是 `x_1` 和 `x_global`
- 说明作者默认“第二层局部表示”更适合和全局表示做分布对齐

### 8. 为什么 4 个模型版本可以一起看

`Diff_DTA_GIN / GCN / GAT / SAGE` 这四个类的差异，主要只在图卷积算子：

- `DenseGINConv`
- `DenseGCNConv`
- `DenseGATConv`
- `DenseSAGEConv`

其余大框架基本不变：

1. 两层局部 `diff_pool`
2. 一条全局图分支
3. 一条蛋白 CNN 分支
4. 四路注意力融合
5. 同样的预测头

所以读法上最省时间的是：

1. 先完全读懂 `Diff_DTA_GIN`
2. 再把其他三个模型当作“卷积算子替换版”

不要四个类一起逐行看，不然信息会重复很多。

## 训练脚本里最值得先看的地方

### 1. 模型到底返回什么

训练时模型不是只返回预测值。

`train()` 里可以看到：

```python
output, l_loss, e_loss, a, x_local, x_global = model(data)
```

说明 `forward()` 返回了 6 个东西：

- `output`
  - 最终预测值
- `l_loss`
  - `diff_pool` 的 link prediction loss
- `e_loss`
  - `diff_pool` 的 entropy loss
- `a`
  - 四路融合注意力权重
- `x_local`
  - 局部分支表示，后面给 `MMDLoss`
- `x_global`
  - 全局分支表示，后面给 `MMDLoss`

### 2. 总损失怎么组成

回归和分类都不是只优化主任务损失。

总损失是：

```python
loss_all = loss + 0.05 * l_loss + 0.05 * e_loss + 0.05 * cl_loss
```

其中：

- `loss`
  - 回归任务用 `MSELoss`
  - 分类任务用 `BCEWithLogitsLoss`
- `l_loss + e_loss`
  - 来自 `dense_diff_pool`
- `cl_loss`
  - 来自 `MMDLoss(x_local, x_global)`

所以如果你后面要做消融实验，最自然的切法就是：

1. 去掉 `MMDLoss`
2. 去掉全局分支
3. 去掉某一层局部池化
4. 去掉四路注意力融合，改回简单拼接

### 3. 怎么选最好模型

`davis/kiba`：

- 每个 epoch 都在测试集上评估
- 用最小 `mse` 保存模型

`Human/Celegans`：

- 在验证集上挑最好 `AUROC`
- 然后去测试集报结果

这意味着两个任务的“最佳模型选择逻辑”不完全一样，后面写实验部分时要注意分开说。

## `model.py` 应该怎么读

### 第一步：只看一个版本

先只看 `Diff_DTA_GIN`。

原因很简单：

- 它最完整
- 结构最清楚
- 另外三个版本只是把图卷积层换成了 `GCN/GAT/SAGE`

等你把 `Diff_DTA_GIN` 看明白，再扫另外三个类，会快很多。

### 第二步：按“主模块”分块看

`model.py` 可以拆成这几块：

#### 1. 注意力模块

- `SelfAttention`
- `Attention_Fusion`

作用：

- `SelfAttention` 用在图分支内部，做表示增强
- `Attention_Fusion` 给四路表示分配权重

#### 2. 局部图分支

例如：

- `GNN_GIN_local`
- `gnn1_pool`
- `gnn1_embed`
- `gnn2_pool`
- `gnn2_embed`

作用：

- 做两层层次化池化
- 提取不同尺度的药物结构表示

#### 3. 全局图分支

例如：

- `GNN_GIN_global`
- `global_gnn_pool`
- `global_gnn`

作用：

- 从另一条路径学习全局药物表示

#### 4. 蛋白分支

关键层：

- `embedding_xt`
- `self.convs`
- `fc1_xt`

作用：

- 蛋白序列先嵌入
- 再走多尺度卷积核的 TextCNN
- 最后压成 128 维表示

#### 5. 融合与预测头

关键层：

- `attention`
- `fc1`
- `fc2`
- `out`

作用：

- 把四路表示融合
- 通过 MLP 输出最终结果

## 数据是怎么流进模型的

这部分建议结合 `utils.py` 和数据构造脚本一起看。

### `TestbedDataset` 在做什么

`utils.py` 里的 `TestbedDataset` 本质上是在把一条样本组织成 PyG 的 `Data`：

- `x`
  - 药物图节点特征
- `edge_index`
  - 药物图边
- `y`
  - 标签
- `target`
  - 蛋白序列编码
- `c_size`
  - 分子原子数

后面再经过 `ToDense(...)`，稀疏图会转成稠密表示，所以训练时模型里直接读：

- `data.x`
- `data.adj`
- `data.target`
- `data.mask`

### 为什么 `kiba` 要拆成两份

这是读代码时很容易忽略的点。

在 `create_data_davis_kiba.py` 里，`kiba` 会被按原子数切成两份：

- `<= 46` 的一份
- `> 46` 的一份

原因是这个项目用了 `ToDense(...)`，而稠密图要求固定节点上限。

所以：

- 常规数据用 `ToDense(46)`
- `kiba` 中较大的分子单独用 `ToDense(268)`

训练时再把两部分结果拼回去。

这就是为什么 `training_davis_kiba.py` 里会有：

- `train_data` / `test_data`
- `train_data1` / `test_data1`

如果你后面改数据处理，这里必须一起改。
