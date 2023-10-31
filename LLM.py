import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from config import LLM_PATH, LEVEL_TOKEN_FORMAT, LABEL_TOKEN_FORMAT
from utils import ADNI


def getLLM() -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModel.from_pretrained(LLM_PATH)
    return model, tokenizer


def tokenizer_add_new_tokens(tokenizer: transformers.PreTrainedTokenizer, token_format: str,
                             token_datas: List[str]) -> List[str]:
    new_tokens = [token_format.format(token_data) for token_data in token_datas]
    tokenizer.add_tokens(new_tokens)
    assert all(token in tokenizer.get_added_vocab() for token in new_tokens), "Tokenizer False Added!"
    return new_tokens


if __name__ == '__main__':
    model, tokenizer = getLLM()
    data, label, labelmap = ADNI.discrete()
