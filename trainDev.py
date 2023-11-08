import torch.utils.data
import LLM
import config
import customdataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from config import *
import utils
from promptGenerateDev import PromptGenerateDev
from maskInfo import maskmodel
from datastruct import split_train_valid_test

from utils import logConfig
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def batchProcess(dataloader: torch.utils.data.dataloader.DataLoader, maskModel: maskmodel,
                 promptmodel: PromptGenerateDev, optimizer: torch.optim.Optimizer,
                 model: PreTrainedModel, tokenizer: PreTrainedTokenizer, criterion: torch.nn.Module,
                 config: trainConfig, correct_predictions: int, total_samples: int, epoch_loss: float, train: bool
                 ):
    for batch in dataloader:
        data = batch['data']
        label = batch['label']
        shape = data.shape

        data = data.to(config.device)
        label = label.to(config.device)

        data_length = shape[-1]
        prompt, mask_pos = promptmodel(data_length, tokenizer, data)
        attention_mask = torch.ones_like(prompt, device=config.device)
        # print(type(model))
        if (len(shape) == 2):
            with torch.no_grad():
                out = model(prompt, attention_mask)
            mask_data = out.last_hidden_state[:, mask_pos.item(), :]
            output = maskModel(mask_data)
        elif (len(shape) == 3):
            # print(attention_mask.shape)
            mask_info_list = []
            for i in range(shape[0]):

                with torch.no_grad():
                    # print(prompt[i, :].shape)
                    # print(attention_mask[i, :].shape)
                    # print(tokens_type_id.shape)
                    promptitem = prompt[i, :]
                    attention_mask_item = attention_mask[i, :]
                    assert promptitem.shape == attention_mask_item.shape
                    # out = model(promptitem, attention_mask_item, tokens_type_id)
                    out = model(prompt[i, :], attention_mask[i, :])
                # print(out)
                mask_item_list = []
                # print(out.last_hidden_state.shape)
                # print(shape)
                # print(mask_pos.shape)
                for j in range(shape[1]):
                    # print(mask_pos[j].item())
                    # print(mask_pos[j].item())
                    mask_item = out.last_hidden_state[j, mask_pos[j].item(), :]
                    mask_item_list.append(mask_item)
                mask_data = torch.stack(mask_item_list, dim=0)
                mask_info = maskModel(mask_data)
                mask_info_list.append(mask_info)
            output = torch.stack(mask_info_list, dim=0)
        # print(output.shape)
        output = output.view(output.shape[0], output.shape[-1])
        loss = criterion(output, label.long())
        # print(loss)
        if (train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算准确率
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

        # 累积epoch损失
        epoch_loss += loss.item()
    return correct_predictions, epoch_loss, total_samples


def encodeData(data: torch.Tensor, tokenizer: PreTrainedTokenizer):
    shape = data.shape
    if (len(shape) == 2):
        encode_tensor_list = []
        for i in range(shape[0]):
            encode_item_list = []
            for j in range(shape[1]):
                encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j])
                encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                encode_item_list.append(encode_item)
            # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
            encode_tensor_list.append(encode_item_list)
        return torch.tensor(encode_tensor_list, dtype=torch.long)
    elif (len(shape) == 3):
        encode_tensor_tensor_list = []
        for i in range(shape[0]):
            encode_tensor_list = []
            for j in range(shape[1]):
                encode_item_list = []
                for k in range(shape[2]):
                    encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j, k])
                    encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                    encode_item_list.append(encode_item)
                # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
                encode_tensor_list.append(encode_item_list)
            encode_tensor_tensor_list.append(encode_tensor_list)
        return torch.tensor(encode_tensor_tensor_list, dtype=torch.long)
    else:
        raise NotImplementedError("len(shape)!=2 or len(shape)!=3 Not Implement!")


def trainDev(datastruct: datastruct, config: config.trainConfig):
    utils.setup_seed(config.seed)
    model, tokenizer = LLM.getLLM()
    data, label, labelmap = datastruct.discrete(config.slice_num)
    for random_state in config.random_state:
        logger = logConfig(config, random_state)

        traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_train_valid_test(data, label,
                                                                                                   randomstate=random_state)

        prompt_num = 1 if len(data.shape) == 2 else data.shape[1]

        promptModel = PromptGenerateDev(config.init_shape, config.emb_dim, config.embLength, config.output_length,
                                        config.device, prompt_num)

        hidden_size = model.config.hidden_size
        if (len(data.shape) == 2):
            maskModel = maskmodel(hidden_size, config.hidden_features, len(labelmap), config.dropout, hidden_size,
                                  config.gru_output_size, prompt_num, need_gru=False)
        elif (len(data.shape) == 3):
            maskModel = maskmodel(hidden_size, config.hidden_features, len(labelmap), config.dropout, hidden_size,
                                  config.gru_output_size, prompt_num, need_gru=True)
            # print(maskModel)
        else:
            raise NotImplementedError()

        for param in model.parameters():
            param.requires_grad = False

        promptModel.to(config.device)
        model.to(config.device)
        maskModel.to(config.device)

        traindataset = customdataset.CustomDataset(traindata, trainlabel)
        validdataset = customdataset.CustomDataset(validdata, validlabel)

        traindataset.data = encodeData(traindataset.data, tokenizer)
        validdataset.data = encodeData(validdataset.data, tokenizer)

        traindataloader = torch.utils.data.dataloader.DataLoader(traindataset, shuffle=True,
                                                                 batch_size=config.batch_size)
        validdataloader = torch.utils.data.dataloader.DataLoader(validdataset, batch_size=config.batch_size)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(list(promptModel.parameters()) + list(maskModel.parameters()), lr=config.lr,
                                      weight_decay=config.weight_decay)

        best_valid_loss = float('inf')
        for epoch in range(1, config.num_epochs + 1):
            # 重置计数器和累积值
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            valid_epoch_loss = 0.0
            valid_correct_predictions = 0
            total_valid_samples = 0

            correct_predictions, epoch_loss, total_samples = batchProcess(traindataloader, maskModel, promptModel,
                                                                          optimizer, model, tokenizer, criterion,
                                                                          config, correct_predictions, total_samples,
                                                                          epoch_loss, True)

            promptModel.eval()
            maskModel.eval()

            valid_correct_predictions, valid_epoch_loss, total_valid_samples = batchProcess(traindataloader, maskModel,
                                                                                            promptModel,
                                                                                            optimizer, model, tokenizer,
                                                                                            criterion,
                                                                                            config,
                                                                                            valid_correct_predictions,
                                                                                            total_valid_samples,
                                                                                            valid_epoch_loss, False)

            promptModel.train()
            maskModel.train()

            # 计算epoch平均损失和准确率
            epoch_loss /= len(traindataloader)
            accuracy = correct_predictions / total_samples
            valid_epoch_loss /= len(validdataloader)
            valid_accuracy = valid_correct_predictions / total_valid_samples

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                # 保存模型
                torch.save(promptModel, f'checkpoint/promptModel/{config.name}_{random_state}_promptModel.pt')
                torch.save(maskModel, f'checkpoint/maskModel/{config.name}_{random_state}_maskModel.pt')
                logger.info(f'Epoch [{epoch}/{config.num_epochs}], , Valid Loss: {valid_epoch_loss:.4f}!')

            logger.info(
                f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')


if __name__ == '__main__':
    # trainDev(config.ADNI, ADNIconfig)
    # trainDev(config.PPMI, PPMIconfig)
    trainDev(config.ADNI_fMRI, config.ADNI_fMRIconfig)
