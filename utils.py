from _import import *
from datastruct import datastruct
from torch.utils.data import Dataset


def setup_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample


ADNI = datastruct('ADNI', 'ADNI')
PPMI = datastruct('PPMI', 'PPMI')
ADNI_fMRI = datastruct('ADNI_fMRI', 'ADNI_90_120_fMRI')
OCD_fMRI = datastruct('OCD_fMRI', 'OCD_90_200_fMRI')
FTD_fMRI = datastruct('FTD_fMRI', 'FTD_90_200_fMRI')

if __name__ == '__main__':
    data, label, labelmap = ADNI.discrete()
    print(type(data[0]))
    print(type(label[0]))
    dataset = CustomDataset(data, label)
    from torch.utils.data.dataloader import DataLoader

    testDataLoader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in testDataLoader:
        print(type(batch['data']))
        print(type(batch['label']))
        print(batch['data'].shape)
        print(batch['label'].shape)
