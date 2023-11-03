import torch
import torch.nn as nn


class maskmodel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,dropout):
        """
        初始化神经网络模型。

        Args:
            in_features (int): 输入特征的数量。
            hidden_features (list): 隐藏层特征的列表。
            out_features (int): 输出特征的数量。

        Returns:
            None
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.dropout=dropout
        self.fc = self._make_fc_layers()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)  # 初始化 Batch Normalization 的权重为1
                nn.init.constant_(m.bias.data, 0)  # 初始化 Batch Normalization 的偏置为0

    def _make_fc_layers(self):
        """
        创建全连接层的私有辅助函数。

        Returns:
            nn.Sequential: 包含全连接层的序列模型。
        """
        layers = []
        in_features = self.in_features
        hidden_features = self.hidden_features

        # 构建残差连接层
        for hidden_feature in hidden_features:
            layers.append(nn.Linear(in_features, hidden_feature))
            layers.append(nn.BatchNorm1d(hidden_feature,eps=1e-18))
            layers.append(nn.Dropout(p=self.dropout))  # 随机失活层
            layers.append(nn.ReLU())  # 激活函数
            in_features = hidden_feature

        # 输出层
        layers.append(nn.Linear(in_features, self.out_features))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.fc(x)
