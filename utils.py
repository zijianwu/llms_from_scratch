import torch


def generate_text_simple(model: torch.nn.Module, 
                         idx: torch.Tensor, 
                         max_new_tokens: int, 
                         context_size: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # shape: [batch_size, context_size, vocab_size]

        logits = logits[:, -1, :]  # predictions for the last token
        probas = torch.softmax(logits, dim=-1)
        new_token = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat([idx, new_token], dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
