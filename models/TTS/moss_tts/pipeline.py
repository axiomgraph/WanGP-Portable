import math
import os
import time
from contextlib import ExitStack
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm
from transformers import AutoTokenizer
from mmgp import runtime, cuda_graphs
from shared.utils.phase_progress import generation_progress, progress_context, PhaseProgress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from .loading import load_transformer, load_codec
from .prompt import build_prompt, audio_codes, sample
from .transformer import GenerationCancelled


class MossTTSPipeline:
    sample_rate = 24000
    frame_length = 1920

    def __init__(self, transformer_path, codec_path, tokenizer_path, features, local=False):
        self.features = features  # the LM decoder optimizations (LMFeatures)
        self.local = local
        self.channels = 2 if local else 1
        self.sample_rate = 48000 if local else 24000
        self.frame_length = 3840 if local else 1920
        self.transformer = load_transformer(transformer_path, features, local=local)
        self.vae = load_codec(codec_path, local=local)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self._interrupt = self._early_stop = False
        self.transformer.abort_requested = lambda: self._interrupt
        self.vae.get_VAE_tile_size = self.get_vae_tile_size
        self.reference_cache = TextEncoderCache()
        self.last_codes = self.last_finish_reason = None
        tqdm.write(f"[MOSS-TTS] {'Local v1.5 4B' if local else 'v1.5 8B'}")

    @staticmethod
    def get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision):
        # Audio frames (80 ms), with the released codec's 10-second causal state.
        return 0 if vae_config == 1 else (25 if vae_config == 3 or (vae_config == 0 and device_mem_capacity < 12000) else 100)

    def check_abort(self, *args):
        if self._interrupt:
            raise GenerationCancelled()

    def request_early_stop(self):
        self._early_stop = True

    def release_cache(self):
        cuda_graphs.invalidate()
        self.transformer.release_cache()
        self.transformer.layer_completed = None
        self.vae._stop_streaming()

    def reference_codes(self, path, tile):
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns, tile)
        def encode(paths):
            set_phase_status("Encoding Voice Reference")
            audio, sr = sf.read(paths[0], dtype="float32", always_2d=True)
            if self.channels == 2:
                if audio.shape[1] == 1:
                    audio = audio.repeat(2, axis=1)
                elif audio.shape[1] != 2:
                    raise ValueError("MOSS Local voice references must be mono or stereo")
                waveform = torch.from_numpy(audio.T.copy())[None]
            else:
                waveform = torch.from_numpy(audio.mean(axis=1).copy())[None, None]
            if sr != self.sample_rate:
                with torch.device(waveform.device):
                    waveform = AF.resample(waveform, sr, self.sample_rate)
            dbfs = 10 * torch.log10(waveform.square().mean() + 1e-9)
            waveform *= 10 ** (max(-3, min(3, float(-20 - dbfs))) / 20)
            samples = waveform.shape[-1]
            chunk = tile * self.frame_length if tile else samples
            parts = []
            runtime.ensure_loaded(self.vae)
            with self.vae.stream(encode=True) if tile and samples > chunk else ExitStack():
                total = math.ceil(samples / chunk)
                with PhaseProgress(total, "Encoding Voice Reference", "tiles", next_status="Encoding Text Prompt") as progress, tqdm(total=total, desc="Encoding Voice Reference", unit="tile", mininterval=1/3) as bar:
                    for start in range(0, samples, chunk):
                        self.check_abort()
                        value = waveform[..., start:start + chunk].to(runtime.execution_device(self.vae))
                        codes = self.vae.encode_tile(value)
                        parts.append(codes[:, 0].T.contiguous().cpu())
                        del codes, value
                        progress.advance(); bar.update()
            return [torch.cat(parts)]
        return self.reference_cache.encode(encode, [path], cache_keys=[key])[0]

    def generate_codes(self, prompt, max_frames, temperature, top_k, top_p, callback):
        if self.local:
            from .local_generation import generate_local_codes
            return generate_local_codes(self, prompt, max_frames, temperature, top_k, top_p, callback)
        model, cfg = self.transformer, self.transformer.config
        n = cfg.n_vq
        # Includes the delay tail and control tokens. References and script also consume context.
        max_steps = max_frames + n + 64
        capacity = len(prompt) + max_steps
        if capacity > cfg.language_config.max_position_embeddings:
            raise ValueError("MOSS-TTS text, reference and duration exceed the model's context capacity")
        model.setup_cache(capacity)
        device = runtime.execution_device(model)
        tokens = torch.full((1, capacity, n + 1), cfg.audio_pad_code, device=device, dtype=torch.long)
        tokens[0, :len(prompt)] = prompt.to(device)
        positions = torch.arange(capacity, device=device)
        prefill_size = 256
        total_layers = math.ceil(len(prompt) / prefill_size) * len(model.language_model.layers)
        set_phase_status("Encoding Text Prompt")
        with PhaseProgress(total_layers, "Encoding Text Prompt", "layers", next_status="Generating Audio Tokens") as progress, tqdm(total=total_layers, desc="Encoding Text Prompt", unit="layer", mininterval=1/3) as bar:
            def completed():
                progress.advance(); bar.update()
            model.layer_completed = completed
            try:
                for start in range(0, len(prompt), prefill_size):
                    end = min(start + prefill_size, len(prompt))
                    logits = model.step(tokens[:, start:end], positions[start:end], end)
            finally:
                model.layer_completed = None
        in_audio, audio_length, delayed, generated = False, 0, None, 0
        self.last_finish_reason = "length"
        forbidden = torch.tensor([cfg.pad_token_id, cfg.audio_assistant_gen_slot_token_id, cfg.audio_assistant_delay_slot_token_id, cfg.audio_end_token_id], device=device)
        audio_text_mask = torch.ones(cfg.language_config.vocab_size, device=device, dtype=torch.bool)
        audio_text_mask[[cfg.audio_assistant_gen_slot_token_id, cfg.audio_assistant_delay_slot_token_id]] = False
        if callback is not None:
            callback(step_idx=-1, override_num_inference_steps=max_frames, progress_title="Generating Audio Tokens", progress_unit="frames")
        updated, started, completed_frames = 0.0, time.perf_counter(), 0
        with tqdm(total=max_frames, desc="Generating Audio Tokens", unit="frame", mininterval=1/3) as bar, cuda_graphs.scope(enabled=self.features.graphs):
            for step in range(max_steps):
                self.check_abort()
                if self._early_stop:
                    self.last_finish_reason = "early_stop"
                    break
                main = logits[0].clone()
                if not in_audio:
                    main[:, forbidden] = -torch.inf
                if in_audio:
                    main.masked_fill_(audio_text_mask, -torch.inf)
                if step == 0:
                    main[:, cfg.audio_assistant_delay_slot_token_id] = -torch.inf
                if step <= n:
                    main[:, cfg.im_end_token_id] = -torch.inf
                if delayed is not None:
                    text_token = cfg.audio_assistant_delay_slot_token_id if delayed < n else cfg.audio_end_token_id
                elif in_audio and audio_length >= max_frames:
                    text_token = cfg.audio_assistant_delay_slot_token_id
                else:
                    text_token = int(sample(main, 1.5, 50, 1.0).item())
                row = tokens[:, len(prompt) + step]
                row[:, 0] = text_token
                active = [head for head in range(n) if audio_length > head and (delayed is None or head > delayed - 1)]
                if active:
                    # Upstream samples codebook 0 separately from the remaining codebooks.
                    if active[0] == 0:
                        score = logits[1].clone(); score[:, -1] = -torch.inf
                        row[:, 1] = sample(score, temperature, top_k, top_p)
                        active = active[1:]
                    if active:
                        score = torch.stack([logits[head + 1][0] for head in active])
                        score[:, -1] = -torch.inf
                        row[0, torch.tensor(active, device=device) + 1] = sample(score, temperature, top_k, top_p)
                generated += 1
                if text_token == cfg.audio_start_token_id:
                    in_audio = True
                if text_token in (cfg.audio_start_token_id, cfg.audio_assistant_gen_slot_token_id, cfg.audio_assistant_delay_slot_token_id):
                    audio_length += 1
                if text_token == cfg.audio_end_token_id:
                    in_audio, audio_length = False, 0
                if text_token == cfg.audio_assistant_delay_slot_token_id and delayed is None:
                    delayed = 0
                if delayed is not None:
                    delayed += 1
                    if delayed > n:
                        delayed = None
                if text_token == cfg.im_end_token_id:
                    self.last_finish_reason = "eos"
                    break
                finished = min(max_frames, max(completed_frames, generated - n - 1))
                bar.update(finished - completed_frames); completed_frames = finished
                now = time.monotonic()
                if callback is not None and now - updated >= 1/3:
                    callback(step_idx=finished - 1, override_num_inference_steps=max_frames, progress_unit="frames", progress_title="Generating Audio Tokens")
                    updated = now
                del logits
                position = len(prompt) + step
                logits = model.step(row[:, None], positions[position:position + 1], position + 1)
        result = audio_codes(tokens[0, len(prompt):len(prompt) + generated].cpu(), cfg)[:, :max_frames]
        if result.shape[-1] == 0 and not self._early_stop:
            raise RuntimeError("MOSS-TTS stopped without producing a complete audio frame")
        tqdm.write(f"[MOSS-TTS] {result.shape[-1]} Audio Frames in {time.perf_counter() - started:.2f}s; Finish: {self.last_finish_reason}")
        return result

    def decode(self, codes, tile):
        set_phase_status("Decoding Speech")
        count = codes.shape[-1]
        chunk = tile or count
        total = math.ceil(count / chunk)
        audio = torch.empty(self.channels, count * self.frame_length, device="cpu", dtype=torch.float32)
        runtime.ensure_loaded(self.vae)
        with self.vae.stream() if tile and count > tile else ExitStack():
            with PhaseProgress(total, "Decoding Speech", "tiles") as progress, tqdm(total=total, desc="Decoding Speech", unit="tile", mininterval=1/3) as bar:
                for start in range(0, count, chunk):
                    self.check_abort()
                    end = min(count, start + chunk)
                    value = codes[:, None, start:end].to(runtime.execution_device(self.vae))
                    decoded = self.vae.decode_tile(value)
                    audio[:, start * self.frame_length:end * self.frame_length] = decoded[0].float().cpu()
                    del decoded, value
                    progress.advance(); bar.update()
        return audio

    @torch.inference_mode()
    @generation_progress
    def generate(self, input_prompt, *, alt_prompt="", audio_guide=None, audio_guide2=None, audio_guide3=None, audio_prompt_type="", duration_seconds=30, model_mode=0,
                 temperature=1.7, top_k=25, top_p=0.8, seed=-1, pause_seconds=0.2, VAE_tile_size=100, callback=None, set_progress_status=None, **kwargs):
        self._interrupt = self._early_stop = False
        self.last_codes = self.last_finish_reason = None
        if seed >= 0:
            torch.manual_seed(seed)
        from shared.utils.speech_dialogue import parse_dialogue, SPEAKER_LINE_PATTERN
        mode = "".join(letter for letter in "ABD" if letter in audio_prompt_type)
        if mode not in ("", "A", "AB", "ABD"):
            raise ValueError("Choose New Voice, One, Two or Three Voice References")
        count = len(mode)
        guides = [audio_guide, audio_guide2, audio_guide3][:count]
        for index, guide in enumerate(guides, 1):
            if not guide:
                raise ValueError(f"Speaker {index} reference audio is required")
        if count > 1:
            turns = parse_dialogue(input_prompt, count, line_labels=True, canonical_ids=True)
        elif SPEAKER_LINE_PATTERN.search(input_prompt):
            turns = parse_dialogue(input_prompt, 1, line_labels=True, canonical_ids=True)
        else:
            if not input_prompt.strip():
                raise ValueError("Enter the words to speak")
            turns = [(0, input_prompt)]
        try:
            with ExitStack() as stack:
                stack.enter_context(runtime.execution_scope(self.check_abort))
                from .modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer
                for module in self.vae.modules():
                    if isinstance(module, MossAudioTokenizerTransformerLayer):
                        stack.callback(module.register_forward_hook(self.check_abort).remove)
                maximum_samples = round(duration_seconds * self.sample_rate)
                maximum = max(1, maximum_samples // self.frame_length)
                pause_samples = round(pause_seconds * self.sample_rate)
                speaker_changes = sum(left[0] != right[0] for left, right in zip(turns, turns[1:]))
                target_frames = max(1, (maximum_samples - pause_samples * speaker_changes) // self.frame_length)
                written, code_parts = 0, []
                audio = torch.empty(self.channels, maximum_samples, device="cpu", dtype=torch.float32)
                total_chars = sum(len(text) for _, text in turns)
                for turn_index, (speaker, text) in enumerate(turns):
                    self.check_abort()
                    pause = pause_samples if turn_index > 0 and speaker != turns[turn_index - 1][0] else 0
                    remaining = (maximum_samples - written - pause) // self.frame_length
                    if self._early_stop or remaining <= 0:
                        break
                    with progress_context(f"Generating Speech - Turn {turn_index + 1}/{len(turns)}, Speaker {speaker + 1}") as progress_callback:
                        reference = self.reference_codes(guides[speaker], VAE_tile_size) if guides else None
                        target = max(1, round(target_frames * len(text) / total_chars)) if int(model_mode) == 1 else None
                        prompt = build_prompt(self.tokenizer, self.transformer.config, text, alt_prompt.strip(), reference, target, local=self.local)
                        def turn_callback(step_idx, **event):
                            event["override_num_inference_steps"] = maximum
                            event["force_refresh"] = step_idx < 0
                            progress_callback(step_idx=(written + pause) // self.frame_length + step_idx, **event)
                        codes = self.generate_codes(prompt, remaining, temperature, top_k, top_p, turn_callback if progress_callback else None)
                        if codes.shape[-1] == 0:
                            break
                        self.release_cache()
                        decoded = self.decode(codes, VAE_tile_size)
                    audio[:, written:written + pause].zero_()
                    written += pause
                    end = written + decoded.shape[-1]
                    audio[:, written:end] = decoded
                    written = end
                    code_parts.append(codes)
                    del decoded
                if not written:
                    return None
                self.last_codes = torch.cat(code_parts, dim=-1)
                audio = audio[:, :written].contiguous()
                if not torch.isfinite(audio).all():
                    raise FloatingPointError("MOSS-TTS decoded non-finite audio")
                return {"x": audio, "audio_sampling_rate": self.sample_rate}
        except GenerationCancelled:
            return None
        finally:
            self.release_cache()

    def release(self):
        self.release_cache()
        self.last_codes = None
        self.transformer = self.vae = None
