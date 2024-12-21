import torch 
import torch.nn as nn 

class SelfAttentionv2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        atten_scores = q @ k.T 
        atten_weights = torch.softmax(atten_scores / k.shape[-1]**0.5, dim=-1)
        context_vecs = atten_weights @ v 
        return context_vecs


# inputs = torch.tensor([
#     [0.43, 0.15, 0.89], 
#     [0.55, 0.87, 0.66], 
#     [0.57, 0.85, 0.64], 
#     [0.22, 0.58, 0.33], 
#     [0.77, 0.25, 0.10], 
#     [0.05, 0.80, 0.55]])

# d_in, d_out = 3, 2
# torch.manual_seed(789)
# sa = SelfAttentionv2(d_in, d_out)
# print(sa(inputs))