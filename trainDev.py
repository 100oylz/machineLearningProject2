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
from cutModel import cutModel
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_length = None


def hook_fn(module, grad_input, grad_output):
    print("Module:", module)
    print("Grad Input:", grad_input)
    print("Grad Output:", grad_output)
    print("\n")


def batchProcess(dataloader: torch.utils.data.dataloader.DataLoader, maskModel: maskmodel,
                 promptmodel: PromptGenerateDev, optimizer: torch.optim.Optimizer,
                 model: PreTrainedModel, tokenizer: PreTrainedTokenizer, criterion: torch.nn.Module,
                 config: trainConfig, correct_predictions: int, total_samples: int, epoch_loss: float, cutmodel:cutModel,train: bool
                 ):
    alpha = 0.1
    for batch in dataloader:
        data = batch['data']
        label = batch['label']
        shape = data.shape

        data = data.to(config.device)
        label = label.to(config.device)

        data_length = shape[-1]

        cls_tensor, data, data_length, mask, mask_tensor, pad_tensor, prompt, sep_tensor, slice,prompt_res = promptmodel(
            data_length, tokenizer, data)

        mask_pos, prompt = promptmodel.finishAdd(cls_tensor, data, data_length, mask, mask_tensor, pad_tensor, prompt,
                                                 sep_tensor,
                                                 slice)
        # print(prompt.requires_grad)
        attention_mask = torch.ones_like(prompt, device=config.device)
        prompt_res=cutmodel(prompt_res)
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
                promptitem = prompt[i, :]
                attention_mask_item = attention_mask[i, :]
                assert promptitem.shape == attention_mask_item.shape
                out = model(prompt[i, :], attention_mask[i, :])
                mask_item_list = []
                for j in range(shape[1]):
                    mask_item = out.last_hidden_state[j, mask_pos[j].item(), :]
                    mask_item_list.append(mask_item)
                mask_data = torch.stack(mask_item_list, dim=0)
                mask_info = maskModel(mask_data)
                mask_info_list.append(mask_info)
            output = torch.stack(mask_info_list, dim=0)
        # print(output.shape)
        output = output.view(output.shape[0], output.shape[-1])
        loss = criterion(output, label.long())

        if(len(shape)==2):
            loss1 =torch.nn.functional.mse_loss(torch.stack([prompt_res]*4),mask_data)
            loss+=alpha*loss1
        # print(loss)
        if (train):
            prompt_init = promptmodel.prompt_init
            mask_slice_init = promptmodel.mask_slice_init

            loss.backward()

            optimizer.step()
            # 检查 prompt_init 是否相同
            if not torch.equal(prompt_init, promptmodel.prompt_init):
                print("prompt_init has been updated!")

            # 检查 mask_slice_init 是否相同
            if not torch.equal(mask_slice_init, promptmodel.mask_slice_init):
                print("mask_slice_init has been updated!")

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
    # model.register_backward_hook(hook_fn)
    model.eval()
    data, label, labelmap = datastruct.discrete(config.slice_num)
    for random_state in config.random_state:
        old_before_prompt, old_after_prompt, old_mask = None, None, None
        logger = logConfig(config, random_state)

        traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_train_valid_test(data, label,
                                                                                                   randomstate=random_state)
        cutmodel=cutModel(config.output_length+2,(128,256,512),model.config.hidden_size)
        # prompt_num = 1 if len(data.shape) == 2 else data.shape[1]
        prompt_num = 1
        promptModel = PromptGenerateDev(config.init_shape, config.emb_dim, config.embLength, config.output_length,
                                        config.device, prompt_num)

        hidden_size = model.config.hidden_size
        if (len(data.shape) == 2):
            maskModel = maskmodel(hidden_size, config.hidden_features, len(labelmap), config.dropout, hidden_size,
                                  config.gru_output_size, prompt_num, need_gru=False)
        elif (len(data.shape) == 3):
            maskModel = maskmodel(hidden_size, config.hidden_features, len(labelmap), config.dropout, hidden_size,
                                  config.gru_output_size, data.shape[1], need_gru=True)
            # print(maskModel)
        else:
            raise NotImplementedError()

        # for param in model.parameters():
        #     param.requires_grad = False

        promptModel.to(config.device)
        cutmodel.to(config.device)
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
        optimizer = torch.optim.NAdam(list(promptModel.parameters()) + list(maskModel.parameters())+list(cutmodel.parameters()), lr=config.lr,
                                      weight_decay=config.weight_decay)

        best_valid_loss = float('inf')
        print(trainlabel.shape)
        print(validlabel.shape)
        print(testlabel.shape)
        for epoch in range(1, config.num_epochs + 1):
            # 重置计数器和累积值
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            valid_epoch_loss = 0.0
            valid_correct_predictions = 0
            total_valid_samples = 0
            print(f'Train!{len(traindataloader)}')
            correct_predictions, epoch_loss, total_samples = batchProcess(traindataloader, maskModel, promptModel,
                                                                          optimizer, model, tokenizer, criterion,
                                                                          config, correct_predictions, total_samples,
                                                                          epoch_loss, cutmodel,True)

            promptModel.eval()
            maskModel.eval()
            print(f"Valid!{len(validdataloader)}")
            valid_correct_predictions, valid_epoch_loss, total_valid_samples = batchProcess(validdataloader, maskModel,
                                                                                            promptModel,
                                                                                            optimizer, model, tokenizer,
                                                                                            criterion,
                                                                                            config,
                                                                                            valid_correct_predictions,
                                                                                            total_valid_samples,
                                                                                            valid_epoch_loss, cutmodel,False)

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
                logger.info(f'Epoch [{epoch}/{config.num_epochs}],Valid Loss: {valid_epoch_loss:.4f}!')
                promptModel.eval()

                beforePrompt, afterPrompt, mask = promptModel.returnPrompt(tokenizer)

                logger.info(f'beforePrompt:{beforePrompt},afterPrompt:{afterPrompt},mask:{mask}')
                promptModel.train()

            logger.info(
                f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')


if __name__ == '__main__':
    trainDev(config.ADNI, ADNIconfig)
    # trainDev(config.PPMI, PPMIconfig)
    # trainDev(config.ADNI_fMRI, config.ADNI_fMRIconfig)
    # trainDev(config.OCD_fMRI, config.OCD_fMRIconfig)
    # trainDev(config.FTD_fMRI, config.FTD_fMRIconfig)
