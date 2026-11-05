"""PTD-specific requirements; input and prompt help comes from Ref2VA."""

PTD_FOLDER = "minimax_h3/ptd_refasymopd"
PTD_HEAD_FILE = "minimax_h3_ref2va_ptd_refasymopd_8step_heads_fp32.safetensors"

PTD_INFOS = """## PTD RefAsymOPD Acceleration

- **Eight-Step Generation:** the accelerator loads automatically. Each video window uses eight Euler steps; step skipping, a second denoising phase and Audio Refinement are unavailable. Do not add another acceleration LoRA.
- **Quality:** published examples focus on image references. The other Ref2VA input modes below remain available, but their generation quality with this accelerator still needs testing.
"""

PTD_DEEPY_INFOS = """PTD RefAsymOPD uses the Ref2VA inputs described below. Use num_inference_steps=8, sample_solver='euler', flow_shift=12, guidance_scale=1, guidance_phases=1 and skip_steps_cache_type='' in every window. Eight built-in adapters activate one per step. No extra acceleration LoRAs, second denoising phase, partial-schedule refinement or Audio Refinement. Published examples cover image references; other input modes await generation-quality validation."""
