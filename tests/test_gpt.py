import torch
from gpt import (
    MultiHeadAttention,
    GELU,
    FeedForward,
    TransformerBlock,
    LayerNorm,
    GPTModel
)


def test_multi_head_attention_shape():
    batch_size, seq_len, d_in, d_out, n_heads = 2, 4, 8, 8, 2
    mha = MultiHeadAttention(d_in, d_out, context_length=seq_len,
                             drop_rate=0.1, n_heads=n_heads, qkv_bias=True)
    x = torch.randn(batch_size, seq_len, d_in)
    out = mha(x)
    assert out.shape == (batch_size, seq_len, d_out), "Output shape mismatch."


def test_gelu_output():
    gelu = GELU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = gelu(x)
    assert out.shape == x.shape, "Output shape mismatch in GELU."
    assert (out >= -1.0).all(), "GELU output seems incorrect."


def test_feedforward_shape():
    cfg = {'emb_dim': 8}
    ff = FeedForward(cfg)
    x = torch.randn(2, 4, 8)
    out = ff(x)
    assert out.shape == x.shape, "FeedForward output shape mismatch."


def test_transformer_block_residual():
    cfg = {
        'emb_dim': 8,
        'context_length': 4,
        'dropout': 0.1,
        'n_heads': 2,
        'qkv_bias': True
    }
    block = TransformerBlock(cfg)
    x = torch.randn(2, 4, 8)
    out = block(x)
    assert out.shape == x.shape, "TransformerBlock shape mismatch."


def test_layer_norm():
    ln = LayerNorm(emb_dim=8)
    x = torch.randn(2, 4, 8)
    out = ln(x)
    assert out.shape == x.shape, "LayerNorm output shape mismatch."
    assert abs(out.mean(dim=-1)).max().item() < 1e-5, "LayerNorm is not centering properly."


def test_gpt_model():
    cfg = {
        'vocab_size': 10,
        'emb_dim': 8,
        'context_length': 4,
        'dropout': 0.1,
        'n_heads': 2,
        'qkv_bias': True,
        'n_layers': 2
    }
    model = GPTModel(cfg)
    x = torch.randint(0, cfg['vocab_size'], (2, cfg['context_length']))
    out = model(x)
    assert out.shape == (2, cfg['context_length'], cfg['vocab_size']), "GPTModel output shape mismatch."