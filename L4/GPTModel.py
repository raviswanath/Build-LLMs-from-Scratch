import torch
import torch.nn as nn
from Transformer import TransformerBlock
from LayerNorm import LayerNorm
import tiktoken


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# torch.manual_seed(123)
# GPT_CONFIG_124M = {
#     "vocab_size": 50257,  # Vocabulary size
#     "context_length": 1024,  # Context length
#     "emb_dim": 768,  # Embedding dimension
#     "n_heads": 12,  # Number of attention heads
#     "n_layers": 12,  # Number of layers
#     "drop_rate": 0.1,  # Dropout rate
#     "qkv_bias": False  # Query-Key-Value bias
# }
#
# tokenizer = tiktoken.get_encoding("gpt2")
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# a = torch.tensor(tokenizer.encode(txt1))
# b = torch.tensor(tokenizer.encode(txt2))
# batch = torch.stack([a, b], dim=0)
# print(batch)
#
# model = GPTModel(GPT_CONFIG_124M)
# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)
