# Adapted from OpenMOSS/MOSS-TTS (Apache-2.0); see LICENSE and NOTICE.
import torch
from .normalizer import normalize_tts_text


def build_prompt(tokenizer, config, text, language="", reference=None, target_tokens=None, local=False):
    """Released user template, including delayed reference codebooks; CPU output."""
    if local:
        return build_local_prompt(tokenizer, config, text, language, reference, target_tokens)
    reference_text = "None"
    if reference is not None:
        token = tokenizer.convert_ids_to_tokens(config.audio_user_slot_token_id)
        reference_text = "[S1]:\n" + tokenizer.convert_ids_to_tokens(config.audio_start_token_id)
        reference_text += token * (reference.shape[0] + config.n_vq - 1)
        reference_text += tokenizer.convert_ids_to_tokens(config.audio_end_token_id)
    content = (f"<user_inst>\n- Reference(s):\n{reference_text}\n- Instruction:\nNone\n- Tokens:\n{target_tokens}"
               f"\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\n{language or None}"
               f"\n- Text:\n{normalize_tts_text(text)}\n</user_inst>")
    content = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    text_ids = torch.tensor(tokenizer.encode(content), device="cpu", dtype=torch.long)
    tokens = torch.full((len(text_ids), config.n_vq + 1), config.audio_pad_code, device="cpu", dtype=torch.long)
    tokens[:, 0] = text_ids
    if reference is not None:
        start = (text_ids == config.audio_start_token_id).nonzero()[0, 0].item() + 1
        for head in range(config.n_vq):
            tokens[start + head:start + head + len(reference), head + 1] = reference[:, head]
    return tokens


def build_local_prompt(tokenizer, config, text, language, reference, target_tokens):
    # The released Local processor tokenizes these pieces separately. Joining the
    # text first changes boundary tokens (notably before punctuation and newlines).
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)
    def rows(ids):
        output = torch.full((len(ids), config.n_vq + 1), config.audio_pad_code, device="cpu", dtype=torch.long)
        output[:, 0] = torch.tensor(ids, device="cpu", dtype=torch.long)
        return output
    prefix = [config.im_start_token_id] + encode("user\n") + encode("<user_inst>\n- Reference(s):\n")
    fields = (f"\n- Instruction:\nNone\n- Tokens:\n{target_tokens}\n- Quality:\nNone"
              f"\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\n{language or None}\n- Text:\n")
    suffix = encode(fields) + encode(text) + encode("\n</user_inst>") + [config.im_end_token_id] + encode("\n")
    suffix += [config.im_start_token_id] + encode("assistant\n") + [config.audio_start_token_id]
    if reference is None:
        return rows(prefix + encode("None") + suffix)
    audio = rows([config.audio_user_slot_token_id] * len(reference))
    audio[:, 1:] = reference
    return torch.cat((rows(prefix), rows([config.audio_start_token_id]), audio, rows([config.audio_end_token_id]), rows(suffix)))


def audio_codes(tokens, config):
    """Recover complete audio frames, including a safely truncated final block."""
    parts, start = [], None
    for index, token in enumerate(tokens[:, 0].tolist()):
        if token == config.audio_start_token_id:
            start = index + 1
        if start is not None and (token == config.audio_end_token_id or index == len(tokens) - 1):
            end = index if token == config.audio_end_token_id else index + 1
            length = end - start - config.n_vq + 1
            if length > 0:
                parts.append(torch.stack([tokens[start + head:start + head + length, head + 1] for head in range(config.n_vq)], dim=0))
            start = None
    if not parts:
        return tokens.new_empty((config.n_vq, 0))
    codes = torch.cat(parts, dim=1)
    if (codes < 0).any() or (codes >= config.audio_vocab_size).any():
        raise RuntimeError("MOSS-TTS produced invalid audio codes")
    return codes


def sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        values, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        logits = torch.full_like(logits, -torch.inf).scatter_(-1, indices, values)
    if top_p < 1:
        probs, order = logits.softmax(-1).sort(descending=True, dim=-1)
        remove = probs.cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits.masked_fill_(torch.zeros_like(remove).scatter_(-1, order, remove), -torch.inf)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
