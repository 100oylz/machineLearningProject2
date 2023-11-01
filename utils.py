import logging

from colorlog import ColoredFormatter

from _import import *
from config import ADNI

from customdataset import CustomDataset


def setup_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def logConfig(config):
    """
    配置日志记录。

    Args:
        config: 配置信息。

    Returns:
        Logger: 配置后的日志记录对象。
    """
    # 创建一个ColoredFormatter
    formatter = ColoredFormatter(
        "%(white)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white,bg_red',
        },
        secondary_log_colors={},
        style='%'
    )

    # 创建新的logger对象
    logger = logging.getLogger(f'{config.name}_logger')
    logger.setLevel(logging.DEBUG)

    # 创建一个用于在控制台输出的处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 创建一个新的文件处理程序（每次都创建新的处理程序）
    log_filename = f'journal/{config.name}.log'
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 移除之前的处理程序（如果有的话）
    for existing_handler in logger.handlers:
        logger.removeHandler(existing_handler)

    # 将处理程序添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



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


