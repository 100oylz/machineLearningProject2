import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from config import *
from utils import logAutoEncoderConfig
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.init as init


class VAE(nn.Module):
    def __init__(self, in_feature: int, features: List[int], out_feature: int, dropout: float):
        super().__init__()
        self.in_feature = in_feature
        self.features = features
        self.dropout = dropout
        self.out_feature = out_feature
        self.encoder_fc_layer = self._make_encoder_fc_layer()
        self.decoder_fc_layer = self._make_decoder_fc_layer()
        self.getmu = nn.Linear(self.features[-1], self.out_feature)
        self.getlogvar = nn.Linear(self.features[-1], self.out_feature)

        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         # 使用He初始化权重，初始化偏置为常数（0）
        #         init.kaiming_normal_(module.weight)
        #         init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.BatchNorm1d):
        #         # 使用He初始化权重，初始化偏置为常数（0）
        #         init.constant_(module.weight, 1)
        #         init.constant_(module.bias, 0)

    def _make_encoder_fc_layer(self):
        layer = []
        in_feature = self.in_feature
        for out_feature in self.features:
            layer.append(nn.Linear(in_feature, out_feature))
            layer.append(nn.BatchNorm1d(out_feature))
            layer.append(nn.Dropout(self.dropout))
            layer.append(nn.PReLU())
            in_feature = out_feature

        return nn.Sequential(*layer)

    def _make_decoder_fc_layer(self):
        layer = []
        in_feature = self.out_feature
        for out_feature in reversed(self.features):
            layer.append(nn.Linear(in_feature, out_feature))
            layer.append(nn.BatchNorm1d(out_feature))
            layer.append(nn.Dropout(self.dropout))
            layer.append(nn.PReLU())
            in_feature = out_feature
        layer.append(nn.Linear(in_feature, self.in_feature))
        layer.append(nn.BatchNorm1d(self.in_feature))
        layer.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layer)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder_fc_layer(x)
        mu = self.getmu(out)
        logvar = self.getlogvar(out)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, x):
        out = self.decoder_fc_layer(x)
        return torch.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


reconstruction_function = nn.MSELoss()


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    KLD_weight = 0.1
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    KLD = KLD.mul_(1.0 / x.shape[0])
    # KL divergence
    return BCE + KLD * KLD_weight, BCE.item(), (KLD * KLD_weight).item()


class VAEdataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.FloatTensor(self.data[item, :])


def trainVAE(dataset: datastruct, config: trainVAEConfig):
    utils.setup_seed(config.seed)
    data, _, _ = dataset.rawdatatonumpy()
    in_feature = data.shape[1] * data.shape[2]
    model = VAE(in_feature, config.features, config.out_feature, config.dropout)
    vaedataset = VAEdataset(data)
    model.to(config.device)
    dataloader = DataLoader(vaedataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.NAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logger = logAutoEncoderConfig(config)
    best_loss = float('inf')
    no_improvement_count = 0
    for epoch in range(1, config.num_epochs + 1):
        total_loss = 0
        total_bce_loss = 0
        total_kld_loss = 0
        for batch in dataloader:
            batch = batch.to(config.device)
            batch = batch.view(batch.shape[0], -1)
            recon_batch, mu, logvar = model(batch)
            loss, bce_loss, kld_loss = loss_function(recon_batch, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bce_loss += bce_loss
            total_kld_loss += kld_loss
        epoch_loss = total_loss / len(dataloader)
        epoch_bce_loss = total_bce_loss / len(dataloader)
        epoch_kld_loss = total_kld_loss / len(dataloader)
        logger.info(f'{config.name}:Epoch [{epoch}/{config.num_epochs}], Loss: {epoch_loss:.4f}')

        # 检查当前epoch的损失是否是最佳损失，如果是则保存模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model, f'checkpoint/AutoEncoder/{config.name}.pt')  # 存储最佳模型权重
            logger.info(
                f'{config.name}:Epoch [{epoch}/{config.num_epochs}]->Saved!Loss:{epoch_loss:.4f},BCE:{epoch_bce_loss},KLD:{epoch_kld_loss}')
            no_improvement_count = 0  # 重置计数器
        else:
            no_improvement_count += 1  # 如果损失没有改善，计数器递增

        if no_improvement_count >= config.enable_num or epoch_loss < config.min_item:
            logger.info(f'{config.name}:Training stopped due to no improvement in loss.')
            break


if __name__ == '__main__':
    trainVAE(ADNI_fMRI, ADNI_fMRI_vaeconfig)
    trainVAE(OCD_fMRI, OCD_fMRI_vaeconfig)
    trainVAE(FTD_fMRI, FTD_fMRI_vaeconfig)
