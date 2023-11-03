import config
import utils
from datastruct import datastruct
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=3, stride=1, padding=1):
        super(EncoderLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=3, stride=1, padding=1):
        super(DecoderLayer, self).__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                                   padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels: List[int], dropout=0.0):
        super(AutoEncoder, self).__init__()
        encoder_layers = [
            EncoderLayer(channels[i], channels[i + 1], dropout=dropout) for i in range(len(channels) - 1)
        ]
        decoder_layers = [
            DecoderLayer(channels[i], channels[i - 1], dropout=dropout) for i in range(len(channels) - 1, 0, -1)
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        # self.encoderlayer = nn.TransformerEncoderLayer(d_model=120, nhead=8, dim_feedforward=256, dropout=dropout)
        # self.decoderlayer = nn.TransformerDecoderLayer(d_model=120, nhead=8, dim_feedforward=256, dropout=dropout)
        # self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=6)
        # self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers=6)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderDataSet(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def train_autoencoder(dataset: datastruct, config: config.trainAutoEncoderConfig):
    logger = utils.logAutoEncoderConfig(config)
    data, label, labelmap = dataset.rawdatatonumpy()
    autoencoder_dataset = AutoEncoderDataSet(data)
    autoencoder = AutoEncoder(config.channels, config.dropout)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(autoencoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    autoencoder.to(config.device)
    dataloader = DataLoader(autoencoder_dataset, batch_size=config.batch_size, shuffle=True)
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(config.device)
            optimizer.zero_grad()  # 梯度清零
            output = autoencoder(data)  # 前向传播
            loss = criterion(output, data)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        logger.info(f'Training Loss: {average_loss:.4f}')

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(autoencoder.state_dict(), f'checkpoint/AutoEncoder/{config.name}.pt')
            logger.info('Saved the best model with average loss: {:.4f}'.format(best_loss))
    print(autoencoder.encoder(data))


if __name__ == '__main__':
    train_autoencoder(config.ADNI_fMRI, config.ADNI_fMRI_autoencoder_config)
