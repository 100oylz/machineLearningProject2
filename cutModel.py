import torch
import torch.nn as nn

class cutModel(nn.Module):
    def __init__(self,in_features,hidden_state_dim,out_features):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.hidden_features=hidden_state_dim
        self.fc=self._make_fc_layers()

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
            layers.append(nn.LayerNorm(hidden_feature, eps=1e-18))
            layers.append(nn.ReLU())  # 激活函数
            in_features = hidden_feature

        # 输出层
        layers.append(nn.Linear(in_features, self.out_features))
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.fc(x)
