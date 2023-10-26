import torch
from transformers import AutoTokenizer, AutoModel
from utils import ADNI

tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
model = AutoModel.from_pretrained("Charangan/MedBERT")

data, label, labelmap = ADNI.discrete()
input_text = f"The patient's features are {data[0, :]}! Is he {labelmap[label[0]]}?"
model.to('cuda')

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    outputs = model(input_ids)

# 获取模型的预测结果（logits）
predictions = outputs['pooler_output']
print(predictions.shape)

# 获取生成的token ID（假设我们想生成下一个词）
predicted_token_id = torch.argmax(predictions).item()

# 将预测的token ID转换为文本
predicted_token = tokenizer.decode(predicted_token_id)

# 输出生成的文本
generated_text = predicted_token
print("生成的文本:", generated_text)