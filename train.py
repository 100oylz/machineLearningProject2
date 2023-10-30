import torch.utils.data

import LLM
import datastruct
from config import *
import utils
from PromptGenerate import PromptGenerate


def train(dataset: datastruct.datastruct, config: trainConfig):
    model, tokenizer = LLM.getLLM()
    data, label, labelmap = dataset.discrete()
    LLM.tokenizer_add_new_tokens(tokenizer, LEVEL_TOKEN_FORMAT, data)
    LLM.tokenizer_add_new_tokens(tokenizer, LABEL_TOKEN_FORMAT, label)
    tokens_num = LLM.get_all_vocab_num(tokenizer)
    promptModel = PromptGenerate(config.init_shape, config.emb_dim, config.embLength, config.output_length)
    for param in model.parameters():
        param.requires_grad = False
    promptModel.to(config.device)
    model.to(config.device)

    for epoch in range(1, config.num_epochs + 1):
        prompt = promptModel(tokens_num)

