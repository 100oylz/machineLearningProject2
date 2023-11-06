import torch.nn as nn
import torch

import config
from datastruct import datastruct


class AutoEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):
        # 编码阶段
        encoder_output = self.transformer_encoder(src)

        # 解码阶段
        decoder_output = self.transformer_decoder(tgt, encoder_output)

        return decoder_output

# TODO:UnFinished!
def train_autoencoder(data: datastruct, config: config.trainAutoEncoderConfig):
    pass



if __name__ == '__main__':
    train_autoencoder(config.ADNI_fMRI, config.ADNI_fMRI_autoencoder_config)
