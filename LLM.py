import torch
import transformers
from transformers import AutoTokenizer, AutoModel, BertConfig
from typing import List, Tuple
from config import LLM_PATH, LEVEL_TOKEN_FORMAT, LABEL_TOKEN_FORMAT, ADNI


def getLLM() -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    获取预训练语言模型和分词器。

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        包含加载的语言模型和分词器的元组。
    """
    custom_config = BertConfig.from_pretrained(LLM_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModel.from_pretrained(LLM_PATH, config=custom_config)
    return model, tokenizer


# 下面的函数目前看起来没有使用，可能是为了将来的用途。
def tokenizer_add_new_tokens(tokenizer: transformers.PreTrainedTokenizer, token_format: str,
                             token_datas: List[str]) -> List[str]:
    """
    向分词器中添加新的标记，并返回添加的标记列表。

    Args:
        tokenizer (transformers.PreTrainedTokenizer): 将添加新标记的分词器。
        token_format (str): 用于创建新标记的格式字符串。
        token_datas (List[str]): 表示要添加的新标记的字符串列表。

    Returns:
        List[str]: 新添加的标记列表。
    """
    new_tokens = [token_format.format(token_data) for token_data in token_datas]
    tokenizer.add_tokens(new_tokens)
    assert all(token in tokenizer.get_added_vocab() for token in new_tokens), "Tokenizer False Added!"
    return new_tokens


if __name__ == '__main__':
    model, tokenizer = getLLM()
    data, label, labelmap = ADNI.discrete()
