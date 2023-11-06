import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


class PromptGenerate(nn.Module):
    def __init__(self, init_shape: tuple = (256, 1), embedding_dim: int = 2048, gru_hidden_size: int = 256,
                 output_length: int = 64, device='cuda'):
        """
        初始化PromptGenerate模型。

        参数：
            init_shape (tuple, optional)：初始形状。默认为(256, 1)。
            embedding_dim (int, optional)：嵌入维度。默认为2048。
            gru_hidden_size (int, optional)：GRU隐藏层维度。默认为256。
            output_length (int, optional)：输出长度。默认为64。
        """
        super().__init__()
        self.init_shape = init_shape
        self.device = device
        temp = torch.randint(0, init_shape[0], init_shape, dtype=torch.long)
        # 注册V，是一个dtype为整形的矩阵，用来计算prompt
        self.register_buffer('V', temp)
        # 注册P，是一个dtype为float的矩阵，用来计算maskpos
        self.register_buffer('P', torch.rand(init_shape))

        self.emb = nn.Embedding(init_shape[0], embedding_dim)
        self.gru = nn.GRU(embedding_dim, gru_hidden_size)
        self.linear = nn.Linear(gru_hidden_size * init_shape[0] * init_shape[1], output_length)
        self.relu = nn.ReLU()
        self.embLength = gru_hidden_size
        self.linear1 = nn.Linear(init_shape[0] * init_shape[1], 2)
        self.to(self.device)

    def forward(self, tokenizer_length: int, datalength: int, tokenizer: PreTrainedTokenizer) -> tuple:
        """
        前向传播函数。

        参数：
            tokenizer_length (int)：标记器的长度。
            datalength (int)：数据长度。
            tokenizer (PreTrainedTokenizer)：分词器。

        返回：
            tuple：包含两个元素的元组，第一个元素为离散化后的整数序列，第二个元素为掩码位置。
        """
        out = self.emb(self.V)
        out, _ = self.gru(out)
        out = self.relu(out)
        out = self.linear(out.view(-1))
        #print('out',out.shape)

        max_vals, _ = torch.max(out, dim=-1, keepdim=True)
        min_vals, _ = torch.min(out, dim=-1, keepdim=True)
        # 得到一个[0,1]区间内的输出，
        normalized_out = (out - min_vals) / (max_vals - min_vals if max_vals != min_vals else 1e-18)

        scaled_integers = normalized_out * tokenizer_length
        # 建立整数映射
        scaled_integers = scaled_integers.clamp(0, tokenizer_length - 1).long().to(self.device)
        #print('1',len(scaled_integers))
        promptlength = len(scaled_integers) + 3

        positem = self.linear1(self.P.view(-1))
        # 建立整数映射
        maskpos = positem[0]
        slicepos = positem[1]


        # maskpos = int(abs(maskpos.item()) * datalength) % datalength
        # print('maskpos11',maskpos)
        # slicepos = int(abs(slicepos.item() * (datalength + 1))) % (datalength + 1)
        # print('slicepos', slicepos)

        maskpos = int(abs(maskpos.item()) * promptlength) % promptlength
        #print('maskpos11', maskpos)
        slicepos = int(abs(slicepos.item() * (promptlength + 1))) % (promptlength + 1)
        #print('slicepos', slicepos)

        masktokenid = tokenizer.mask_token_id
        starttokenid = tokenizer.cls_token_id
        endtokenid = tokenizer.sep_token_id

        start_tensor = torch.Tensor([starttokenid]).to(self.device)
        end_tensor = torch.Tensor([endtokenid]).to(self.device)
        mask_tensor = torch.Tensor([masktokenid]).to(self.device)

        #scaled_integers纬度为67
        scaled_integers = torch.cat(
            (start_tensor, scaled_integers[:maskpos], mask_tensor, scaled_integers[maskpos:], end_tensor))

        #print('scaled_integers',len(scaled_integers))
        beforePrompt = scaled_integers[:slicepos + 1]
        afterPrompt = scaled_integers[slicepos + 1:]

        maskpos = maskpos + 1 + datalength if maskpos >= slicepos else maskpos + 1

        return beforePrompt, afterPrompt, maskpos
