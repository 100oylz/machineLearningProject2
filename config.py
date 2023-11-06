from dataclasses import dataclass
from typing import List, Tuple

from datastruct import datastruct

# MED_BERT = "gpt2-large"
# LLM_PATH = "Charangan/MedBERT"
# LLM_PATH = "C:\\Users\\OYLZ\\.cache\\huggingface\\hub\\models--Charangan--MedBERT\\snapshots\\315cdfc82d4d6eb1cabfb35444095e5b975d4d9d"
LLM_PATH = 'bert-large-uncased'
# LLM_PATH = r"D:\transformermodel\bert-base-uncased"
# LEVEL_TOKEN_FORMAT = '[level_{}]'
# LABEL_TOKEN_FORMAT = '[{}]'
LEVEL_TOKEN_FORMAT = '{:0>2}'
LABEL_TOKEN_FORMAT = '{}'
LABEL_WORD_MAP = ['best', 'good', 'normal', 'bad', 'worst']

DEFAULTMILESTONES = [10, 20]
DEFAULTCHANNELS = [90, 128, 256, 512, 256, 128, 64, 1]


@dataclass
class trainConfig():
    name: str
    milestones: List[int]
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 40
    seed: int = 24
    dropout: float = 0.2
    device: str = 'cuda'
    init_shape: Tuple[int, int] = (32, 32)
    emb_dim: int = 64
    embLength: int = 256
    output_length: int = 64
    batch_size: int = 4
    mask_str: str = "[MASK]"
    slice_num: int = 100
    hidden_features: Tuple[int] = (128, 64)
    random_state: Tuple[int] = (0,1, 2, 3, 4)
    saved_by_valid_loss: bool = True
    dim: int = 2


ADNIconfig = trainConfig('ADNI', milestones=DEFAULTMILESTONES)
PPMIconfig = trainConfig('PPMI', milestones=DEFAULTMILESTONES, batch_size=16, dropout=0.3)
ADNI_fMRIconfig = trainConfig('ADNI_fMRI', milestones=DEFAULTMILESTONES, dim=2, batch_size=4)
OCD_fMRIconfig = trainConfig('OCD_fMRI', milestones=DEFAULTMILESTONES, dim=2, batch_size=2)
FTD_fMRIconfig = trainConfig('FTD_fMRI', milestones=DEFAULTMILESTONES, dim=2, batch_size=2)

ADNI = datastruct('ADNI', 'ADNI')
PPMI = datastruct('PPMI', 'PPMI')
ADNI_fMRI = datastruct('ADNI_fMRI', 'ADNI_90_120_fMRI')
OCD_fMRI = datastruct('OCD_fMRI', 'OCD_90_200_fMRI')
FTD_fMRI = datastruct('FTD_fMRI', 'FTD_90_200_fMRI')
