# 当前路径包
import LLM
# 第三方库
import torch
import torch.nn as nn
from typing import Tuple
from transformers import PreTrainedTokenizer


class PromptGenerateDev(nn.Module):
    """
    生成模型的类，用于生成prompt和mask。

    Args:
        init_shape (Tuple[int, int]): 初始化buffer的形状。
        embedding_dim (int): 嵌入维度。
        gru_hidden_state (int): GRU隐藏状态的维度。
        promptlength (int): 生成的prompt的长度。
        device (str | torch.device): 指定设备（'cpu' 或 torch.device 对象）。
        prompt_num (int): 生成的prompt的数量。
    """

    def __init__(self, init_shape: Tuple[int, int], embedding_dim: int, gru_hidden_state: int, promptlength: int,
                 device: str | torch.device, prompt_num: int):
        super().__init__()
        # 初始化buffer，分别用来进行初始化prompt，以及计算maskpos和slicepos
        prompt_init_data = torch.randint(0, init_shape[0], init_shape, dtype=torch.long)
        mask_slice_init_data = torch.rand(init_shape[0] * init_shape[1])
        self.register_buffer('prompt_init', prompt_init_data)
        self.register_buffer('mask_slice_init', mask_slice_init_data)
        # 计算prompt的网络结构定义,embedding-gru-mlp
        self.emb = nn.Embedding(init_shape[0], embedding_dim)
        self.gru = nn.GRU(embedding_dim, gru_hidden_state)
        self.prompt_num = prompt_num
        self.promptlength = promptlength
        self.generate_prompt = nn.Linear(gru_hidden_state * init_shape[0] * init_shape[1], prompt_num * promptlength)

        self.relu = nn.ReLU()
        # 计算mask_pos和slice_pos的网络结构，mlp
        self.generate_mask_slice = nn.Linear(init_shape[0] * init_shape[1], prompt_num * 2)

        self.device = device
        self.to(self.device)

    def forward(self, data_length: int, tokenizer: PreTrainedTokenizer, data: torch.Tensor):
        """
        前向传播函数，用于生成prompt和mask。

        Args:
            data_length (int): 输入数据的长度。
            tokenizer (PreTrainedTokenizer): 分词器对象。
            data (torch.Tensor): 输入数据张量。

        Returns:
            torch.Tensor: 生成的prompt张量。
            torch.Tensor: 生成的mask张量。
        """
        assert data.dtype == torch.long
        out = self.emb(self.prompt_init)
        out, _ = self.gru(out)
        out = self.relu(out)
        out = self.generate_prompt(out.view(-1))
        prompt = out.view(self.prompt_num, self.promptlength)
        # 标准化确保在0-1区间
        max_vals, _ = torch.max(prompt, dim=1, keepdim=True)
        min_vals, _ = torch.min(prompt, dim=1, keepdim=True)
        slice = max_vals - min_vals
        slice[slice == 0] = 1e-18
        normalized_out = (prompt - min_vals) / slice
        # 进行映射
        prompt = normalized_out * tokenizer.vocab_size
        prompt = prompt.clamp(0, tokenizer.vocab_size - 1).long().to(self.device)
        del slice, normalized_out, out, min_vals, max_vals
        torch.cuda.empty_cache()
        mask_slice = self.generate_mask_slice(self.mask_slice_init)
        mask_slice = mask_slice.view(self.prompt_num, 2)
        # 生成maskpos
        mask = mask_slice[:, 0]
        mask = torch.abs_(mask) * data_length
        mask = mask.clamp(0, data_length).long().to(self.device)
        # 生成slicepos
        slice = mask_slice[:, 1]
        slice = torch.abs_(slice) * (data_length + 1)
        slice = slice.clamp(0, data_length + 1).long().to(self.device)
        del mask_slice
        masktokenid = tokenizer.mask_token_id
        clstokenid = tokenizer.cls_token_id
        septokenid = tokenizer.sep_token_id
        padtokenid = tokenizer.pad_token_id

        mask_tensor = torch.tensor([masktokenid], dtype=torch.long, device=self.device)
        cls_tensor = torch.tensor([clstokenid], dtype=torch.long, device=self.device)
        sep_tensor = torch.tensor([septokenid], dtype=torch.long, device=self.device)
        pad_tensor = torch.tensor([padtokenid], dtype=torch.long, device=self.device)
        assert type(self.prompt_num) == int
        if (self.prompt_num == 1):
            mask, prompt = self.add_mask_slice_data_toPrompt(cls_tensor, data, data_length, mask, mask_tensor, prompt,
                                                             sep_tensor, slice, pad_tensor, True)
        elif (self.prompt_num > 1):
            # 仅限3维数据
            assert len(data.shape) == 3
            # 将其加入每一个时序
            assert self.prompt_num == data.shape[1]

            masklist, promptlist = [], []
            add_cls = True
            for i in range(self.prompt_num):
                maskitem, promptitem = self.add_mask_slice_data_toPrompt(cls_tensor, data[:, i, :], data_length,
                                                                         mask[i], mask_tensor,
                                                                         prompt[i, :], sep_tensor, slice[i], pad_tensor,
                                                                         add_cls)
                masklist.append(maskitem)
                promptlist.append(promptitem)
                if (add_cls == True):
                    add_cls = False
            mask = torch.stack(masklist, dim=0)
            prompt = torch.stack(promptlist, dim=0)
        else:
            raise NotImplementedError("Only For prompt_num>=1 And Type(prompt_num)==int")
        return prompt, mask

    def add_mask_slice_data_toPrompt(self, cls_tensor, data, data_length, mask, mask_tensor, prompt, sep_tensor, slice,
                                     pad_tensor,
                                     add_cls: bool = True):
        """
        将mask、data等信息加入到prompt中，并返回生成的mask和prompt。

        Args:
            cls_tensor (torch.Tensor): CLS标记的张量。
            data (torch.Tensor): 输入数据张量。
            data_length (int): 输入数据的长度。
            mask (torch.Tensor): mask位置的张量。
            mask_tensor (torch.Tensor): MASK标记的张量。
            prompt (torch.Tensor): 当前的prompt张量。
            sep_tensor (torch.Tensor): SEP标记的张量。
            slice (torch.Tensor): 切分位置的张量。
            pad_tensor (torch.Tensor): PAD标记的张量。
            add_cls (bool, optional): 是否加入CLS标记。默认为True。

        Returns:
            torch.Tensor: 生成的mask张量。
            torch.Tensor: 生成的prompt张量。
        """
        prompt = prompt.view(-1)
        maskpos = mask.item()
        slicepos = slice.item()
        # 加入mask
        if (add_cls):
            prompt = torch.cat(
                (cls_tensor, prompt[:maskpos], mask_tensor, prompt[maskpos:], sep_tensor)
            )
        else:
            prompt = torch.cat(
                (prompt[:maskpos], mask_tensor, prompt[maskpos:], sep_tensor, pad_tensor)
            )
        # 切分prompt
        beforeprompt = prompt[:slicepos + 1]
        afterprompt = prompt[slicepos + 1:]
        # 堆叠prompt至特定形状
        beforeprompt = torch.stack([beforeprompt] * data.shape[0], dim=0)
        afterprompt = torch.stack([afterprompt] * data.shape[0], dim=0)
        # 重新计算mask，如果maskpos>=slicepos,则mask+[cls_tensor](1)+data(data_length),反之mask+[cls_tensor](1)
        mask = mask + data_length if maskpos >= slicepos else mask
        if (add_cls):
            mask = mask + 1
        # 加入data
        prompt = torch.cat(
            (beforeprompt, data, afterprompt), dim=1
        )
        return mask, prompt


if __name__ == '__main__':
    llm, tokenizer = LLM.getLLM()
    promptmodeldev = PromptGenerateDev((16, 16), 256, 64, 64, 'cuda', 16)
    data = torch.randint(0, 10, (16, 16, 64)).to('cuda')
    prompt, mask = promptmodeldev(64, tokenizer, data)
    print(prompt)
    print(mask)
