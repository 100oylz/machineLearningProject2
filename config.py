from dataclasses import dataclass

# MED_BERT = "gpt2-large"
# LLM_PATH = "Charangan/MedBERT"
# LLM_PATH = "C:\\Users\\OYLZ\\.cache\\huggingface\\hub\\models--Charangan--MedBERT\\snapshots\\315cdfc82d4d6eb1cabfb35444095e5b975d4d9d"
LLM_PATH='bert-large-uncased'

# LEVEL_TOKEN_FORMAT = '[level_{}]'
# LABEL_TOKEN_FORMAT = '[{}]'
LEVEL_TOKEN_FORMAT = '{:0>2}'
LABEL_TOKEN_FORMAT = '{}'
LABEL_WORD_MAP = ['best', 'good', 'normal', 'bad', 'worst']


@dataclass
class trainConfig():
    name: str
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_epochs: float = 100
    seed: int = 24
    dropout: float = 0.2
    device: str = 'cuda'
    init_shape: tuple[int, int] = (256, 1)
    emb_dim: int = 64
    embLength: int = 256
    output_length: int = 64
    batch_size: int = 16
    mask_str: str = "[MASK]"
    slice_num: int = 50


ADNIconfig = trainConfig('ADNI')
