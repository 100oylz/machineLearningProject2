from dataclasses import dataclass

# MED_BERT = "gpt2-large"
LLM_PATH = "Charangan/MedBERT"
LEVEL_TOKEN_FORMAT = '[level_{}]'
LABEL_TOKEN_FORMAT = '[{}]'


@dataclass
class trainConfig():
    lr: float
    weight_decay: float
    num_epochs: float
    name: str
    seed: int
    dropout: float
    device: str
    init_shape: tuple[int, int]
    emb_dim: int
    embLength: int
    output_length: int
    mask_str: str = "[MASK]"
