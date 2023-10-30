import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from typing import List
from config import LLM_PATH, LEVEL_TOKEN_FORMAT, LABEL_TOKEN_FORMAT
from utils import ADNI


def getLLM() -> (transformers.PreTrainedModel, transformers.PreTrainedTokenizer):
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModel.from_pretrained(LLM_PATH)
    return model, tokenizer


def tokenizer_add_new_tokens(tokenizer: transformers.PreTrainedTokenizer, token_format: str,
                             token_datas: List[str]) -> List[str]:
    new_tokens = [token_format.format(token_data) for token_data in token_datas]
    tokenizer.add_tokens(new_tokens)
    assert all(token in tokenizer.get_added_vocab() for token in new_tokens), "Tokenizer False Added!"
    return new_tokens


def get_all_vocab_num(tokenizer: transformers.PreTrainedTokenizer):
    vocab_size = len(tokenizer.get_vocab())
    special_tokens_mask = len(tokenizer.all_special_tokens)
    added_vocab = len(tokenizer.additional_special_tokens)

    total_vocab_num = vocab_size + special_tokens_mask + added_vocab
    return total_vocab_num


if __name__ == '__main__':
    model, tokenizer = getLLM()
    data, label, labelmap = ADNI.discrete()
    print(get_all_vocab_num(tokenizer))
    level_tokens = tokenizer_add_new_tokens(tokenizer, LEVEL_TOKEN_FORMAT, [str(i) for i in range(ADNI.slicenum)])
    print(get_all_vocab_num(tokenizer))
    label_tokens = tokenizer_add_new_tokens(tokenizer, LABEL_TOKEN_FORMAT, labelmap)
    print(get_all_vocab_num(tokenizer))
