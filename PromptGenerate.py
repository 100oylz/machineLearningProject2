import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers


class PromptGenerate(nn.Module):
    def __init__(self, init_shape=(256, 1), emb_dim=2048, embLength=256, output_length=64):
        super().__init__()
        self.init_shape = init_shape
        temp = torch.randint(0, init_shape[0], init_shape, dtype=torch.long)
        self.register_buffer('V', temp)
        self.register_buffer('P', temp.float())
        self.emb = nn.Embedding(init_shape[0], emb_dim)
        self.gru = nn.GRU(emb_dim, embLength)
        self.linear = nn.Linear(embLength * init_shape[0], output_length)
        self.embLength = embLength
        self.linear1 = nn.Linear(init_shape[0] * init_shape[1], 1)

    def forward(self, tokenizer_length: int, datalength: int, tokenizer: transformers.PreTrainedTokenizer):
        out = self.emb(self.V)
        out, _ = self.gru(out)
        out = self.linear(out.view(-1))  # Fixed dimensionality

        # Normalize values between 0 and 1
        max_vals, _ = torch.max(out, dim=-1, keepdim=True)
        min_vals, _ = torch.min(out, dim=-1, keepdim=True)
        normalized_out = (out - min_vals) / (
            max_vals - min_vals if max_vals != min_vals else 1e-18)  # Added small epsilon to avoid division by zero

        # Scale values to the desired range [0, tokenizer_length]
        scaled_integers = normalized_out * tokenizer_length

        # Convert to long tensor
        scaled_integers = scaled_integers.clamp(0, tokenizer_length - 1).long()  # Ensure values are within range

        # Replace [MASK] token with [UNK] token
        special_token_id = tokenizer.convert_tokens_to_ids('[UNK]')
        scaled_integers[scaled_integers == special_token_id] = special_token_id

        maskpos = self.linear1(self.P.view(-1))
        return scaled_integers, abs(int(maskpos.item() * (datalength + 1))) // (datalength + 1)


if __name__ == '__main__':
    promptModel = PromptGenerate()
    res = promptModel(50000)
    print(res)
    print(res.shape)
