import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                 drop_rate, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, 'd_out must be divisible by n_heads'
        self.head_dim = d_out // n_heads
        self.n_heads = n_heads
        self.d_out = d_out

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask',
                             torch.triu(
                                 torch.ones(context_length, context_length),
                                 diagonal=1))
        self.dropout = nn.Dropout(drop_rate)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_heads, self.head_dim)

        keys = keys.transpose(1, 2)  # (b, n_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # (b, n_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # (b, n_heads, num_tokens, head_dim)

        scores = queries @ keys.transpose(-2, -1)  # (b, n_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        scores.masked_fill_(mask_bool, float('-inf'))

        attn_weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, n_heads, num_tokens, head_dim)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg['emb_dim'])
        self.attn = MultiHeadAttention(
            cfg['emb_dim'], 
            cfg['emb_dim'], 
            cfg['context_length'],
            cfg['drop_rate'], 
            cfg['n_heads'],
            qkv_bias=cfg['qkv_bias'])
        self.dropout_shortcut = nn.Dropout(cfg['drop_rate'])
        self.ln2 = nn.LayerNorm(cfg['emb_dim'])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = shortcut + x
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        self.ln = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = x + pos
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.ln(x)
        logits = self.out_head(x) # shape: [batch_size, seq_len, vocab_size]
        return logits
