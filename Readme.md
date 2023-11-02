# 机器学习前沿项目二

## 小组信息

|      | 姓名     | 学号          | 完成任务                                |
| ---- | -------- | ------------- | --------------------------------------- |
| 组长 | 欧阳林茁 | 2021040907008 | MLP及对应论文内容撰写，论文排版，组织   |
| 组员 | 刘宸博   | 2021150902016 | SVM以及Random Foreset及对应论文内容撰写 |

## 数据集处理

1. 标准化
   - 避免离群数据点对于数据的影响
2. 根据标准化后的数据进行离散化
   - 将连续型数据进行分区离散化，从而使得其适配NLP要求的input_id

### 标准化

#### 二维数据

对于[batch_size,feature_num]的矩阵，取出[:,feature]的数据进行标准化

#### 三维数据

对于[batch_size,times,feature_num]的矩阵，取出[:,:,feature]的数据进行标准化

### 离散化

#### 二维数据

对于[batch_size,feature_num]的矩阵，取出[:,feature]的数据，得到其max和min，在[min,max+eps]中化[左开右闭]的slicenum个空间，然后建立原数据与区间的映射关系

#### 三维数据

对于[batch_size,times,feature_num]的矩阵，取出[:,:,feature]的数据，得到其max和min，在[min,max+eps]中化[左开右闭]的slicenum个空间，然后建立原数据与区间的映射关系

## prompt生成

1. 获取LLM的vocab_size(词表空间)
2. 规定prompt的长度
   - (==也许也可以通过神经网络学习到==)
3. 获取不加mask时得到的prompt+data的数据长度datalength
4. 根据神经网络，分别初始化两个buffer，一个buffer计算prompt，另一个计算maskpos
5. 拼接prompt和data
6. 在拼接后的数据上的maskpos，插入mask

### 二维数据

#### prompt生成

定义buffer P，P经过embedding，进行gru之后进行linear，得到[promptlength]的矩阵，然后对于每一个值进行vocab_size的整数空间映射

#### 计算maskpos

定义buffer V，V直接全连接出一个标量，再对标量进行datalength+1的整数空间映射(==此处只有一层全连接，可能可以通过更加优秀的网络进行计算==)

### 三维数据

==TODO，目前reshape不太可行(会超出LLM的seglength)，估计需要进行encoder精简特征==

### mask信息读取

1. 获取到LLM的last_hidden_state，从中晒出[:,maskpos,:]的矩阵
2. 根据矩阵进行全连接，多分类(==可能可以通过rnn或者cnn进行更好的特征筛选再进行全连接==)

# 训练过程

1. 获取LLM的model和tokenizer
2. 初始化promptModel和maskModel
3. 数据加载
4. 生成prompt
5. 加入LLM，计算output的last_hidden_state
6. 取出mask位置的信息
7. 进行mask信息提取，加入maskModel
8. 计算损失，同步更新promptModel和maskModel







