# Adapted from OpenMOSS/MOSS-TTS (Apache-2.0); see LICENSE and NOTICE.
"""Depth-autoregressive generation for MOSS Local v1.5."""
import math
import time
import torch
from tqdm import tqdm
from mmgp import runtime, cuda_graphs
from shared.utils.phase_progress import PhaseProgress, set_phase_status

def sample(logits, temperature, top_k, top_p):
    """Released Local sampler, including ties at the top-k threshold."""
    if temperature <= 0:
        return logits.argmax(-1)
    scores = logits.float() / temperature
    if 0 < top_k < scores.shape[-1]:
        threshold = scores.topk(top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if 0 < top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter_(-1, indices, remove), -torch.inf)
    return torch.multinomial(scores.softmax(-1), 1).squeeze(-1)


def generate_local_codes(pipe, prompt, max_frames, temperature, top_k, top_p, callback):
    model, cfg = pipe.transformer, pipe.transformer.config
    capacity = len(prompt) + max_frames
    if capacity > cfg.qwen3_config.max_position_embeddings:
        raise ValueError("MOSS Local text, reference and duration exceed the context capacity")
    model.setup_cache(capacity)
    device = runtime.execution_device(model)
    tokens = torch.full((1, capacity, cfg.n_vq + 1), cfg.audio_pad_code, device=device, dtype=torch.long)
    tokens[0, :len(prompt)] = prompt.to(device)
    positions = torch.arange(capacity, device=device)
    depth_positions = torch.arange(13, device=device)
    total = math.ceil(len(prompt) / 256) * len(model.transformer.layers)
    set_phase_status("Encoding Text Prompt")
    with PhaseProgress(total, "Encoding Text Prompt", "layers", next_status="Generating Audio Tokens") as progress, tqdm(total=total, desc="Encoding Text Prompt", unit="layer", mininterval=1/3) as bar:
        def completed():
            progress.advance(); bar.update()
        model.layer_completed = completed
        try:
            for start in range(0, len(prompt), 256):
                end = min(start + 256, len(prompt))
                hidden = model.step(tokens[:, start:end], positions[start:end], end)
        finally:
            model.layer_completed = None
    if callback is not None:
        callback(step_idx=-1, override_num_inference_steps=max_frames, progress_title="Generating Audio Tokens", progress_unit="frames")
    pipe.last_finish_reason = "length"
    generated, updated = 0, 0.0
    started = time.perf_counter()
    with tqdm(total=max_frames, desc="Generating Audio Tokens", unit="frame", mininterval=1/3) as bar, cuda_graphs.scope(enabled=pipe.features.graphs):
        for frame in range(max_frames):
            pipe.check_abort()
            if pipe._early_stop:
                pipe.last_finish_reason = "early_stop"
                break
            local = model.depth_step(hidden, depth_positions[:1])
            if int(sample(model.text_head(local).float(), 1.0, 50, 1.0).item()) == 1:
                pipe.last_finish_reason = "eos"
                break
            row = tokens[:, len(prompt) + frame:len(prompt) + frame + 1]
            row[..., 0] = cfg.audio_assistant_slot_token_id
            for channel in range(cfg.n_vq):
                pipe.check_abort()
                code = sample(model.audio_head(local, channel).float(), temperature, top_k, top_p)
                row[:, 0, channel + 1] = code
                if channel + 1 < cfg.n_vq:
                    local = model.depth_step(model.audio_embedding(code[:, None], channel), depth_positions[channel + 1:channel + 2])
            generated += 1
            bar.update()
            now = time.monotonic()
            if callback is not None and now - updated >= 1/3:
                callback(step_idx=generated - 1, override_num_inference_steps=max_frames, progress_unit="frames", progress_title="Generating Audio Tokens")
                updated = now
            if generated < max_frames:
                pos = len(prompt) + frame
                hidden = model.step(row, positions[pos:pos + 1], pos + 1)
    result = tokens[0, len(prompt):len(prompt) + generated, 1:].T.contiguous().cpu()
    if not generated and not pipe._early_stop:
        raise RuntimeError("MOSS Local stopped without producing an audio frame")
    tqdm.write(f"[MOSS Local] {generated} Audio Frames in {time.perf_counter() - started:.2f}s; Finish: {pipe.last_finish_reason}")
    return result
