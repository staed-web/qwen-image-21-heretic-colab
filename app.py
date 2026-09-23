"""
Qwen-Image-2.1 + Heretic text encoder — Gradio app for Hugging Face Spaces (ZeroGPU-friendly).

Uses transformers bf16 Heretic TE shards (NOT the ComfyUI-only GGUF).
"""

from __future__ import annotations

import os
import random
from functools import lru_cache

import gradio as gr
import torch
from PIL import Image

# ZeroGPU on HF Spaces (optional)
try:
    import spaces  # type: ignore
except ImportError:  # local / Colab / non-Spaces
    spaces = None

BASE_MODEL = os.environ.get("QWEN_IMAGE_BASE", "Qwen/Qwen-Image-2.1")
HERETIC_TE = os.environ.get(
    "QWEN_IMAGE_HERETIC_TE",
    "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic",
)
DTYPE = torch.bfloat16

# Colab / free-tier friendly (official defaults are 2048 — too heavy for T4 / ZeroGPU)
ASPECT_RATIOS = {
    "1:1 (1024x1024)": (1024, 1024),
    "1:1 (768x768)": (768, 768),
    "16:9 (1280x720)": (1280, 720),
    "9:16 (720x1280)": (720, 1280),
    "4:3 (1152x896)": (1152, 896),
    "3:4 (896x1152)": (896, 1152),
    "3:2 (1152x768)": (1152, 768),
    "2:3 (768x1152)": (768, 1152),
}

TRANSPARENCY_PREFIX = "This is an RGBA image with transparency. "
TRANSPARENCY_SUFFIX = " The image has alpha channel and the background is transparent."
MAX_SEED = 2**31 - 1

_pipe = None


def _import_qwen3_vl():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText


def get_pipe():
    """Lazy-load pipeline once (Spaces cold start / ZeroGPU)."""
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import QwenImage21Pipeline

    cls = _import_qwen3_vl()
    print(f"Loading Heretic TE: {HERETIC_TE}")
    text_encoder = cls.from_pretrained(HERETIC_TE, torch_dtype=DTYPE)

    print(f"Loading pipeline: {BASE_MODEL}")
    pipe = QwenImage21Pipeline.from_pretrained(
        BASE_MODEL,
        text_encoder=text_encoder,
        torch_dtype=DTYPE,
    )
    # CPU offload works on Colab T4 and many Spaces GPUs; ZeroGPU still benefits from bf16.
    if os.environ.get("QWEN_IMAGE_NO_OFFLOAD", "").lower() not in ("1", "true", "yes"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    _pipe = pipe
    print("Pipeline ready.")
    return _pipe


def apply_transparency_template(prompt: str, transparent: bool) -> str:
    prompt = (prompt or "").strip()
    if not transparent:
        return prompt
    low = prompt.lower()
    if "rgba image with transparency" in low and "background is transparent" in low:
        return prompt
    return f"{TRANSPARENCY_PREFIX}{prompt}.{TRANSPARENCY_SUFFIX}"


def _generate_impl(
    prompt,
    reference_image,
    aspect_ratio,
    steps,
    seed,
    randomize_seed,
    transparent_rgba,
    negative_prompt,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not str(prompt).strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed or seed is None:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    final_prompt = apply_transparency_template(str(prompt), bool(transparent_rgba))
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    pipe = get_pipe()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(
        prompt=final_prompt,
        negative_prompt=negative_prompt or " ",
        width=width,
        height=height,
        num_inference_steps=int(steps),
        generator=generator,
    )

    if reference_image is not None:
        if not isinstance(reference_image, Image.Image):
            reference_image = Image.fromarray(reference_image)
        kwargs["image"] = reference_image

    with torch.inference_mode():
        out = pipe(**kwargs).images[0]

    return out, seed, final_prompt


if spaces is not None:
    generate = spaces.GPU(duration=180)(_generate_impl)
else:
    generate = _generate_impl


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-2.1 + Heretic TE") as demo:
        gr.Markdown(
            """
            # Qwen-Image-2.1 + Heretic Text Encoder
            Base: `Qwen/Qwen-Image-2.1` | TE: `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic`
            (transformers bf16 shards — **not** the ComfyUI-only GGUF)

            Official blog / demos: [qwen.ai blog](https://qwen.ai/blog?id=qwen-image-2.1) · [wuli.art](https://wuli.art/explore) (CN)
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder='e.g. A neon shop sign that reads "QWEN IMAGE 2.1", rainy night',
                )
                reference_image = gr.Image(
                    label="Optional reference image (edit / I2I)",
                    type="pil",
                    image_mode="RGBA",
                )
                aspect_ratio = gr.Dropdown(
                    label="Aspect ratio (memory-friendly)",
                    choices=list(ASPECT_RATIOS.keys()),
                    value="1:1 (1024x1024)",
                )
                with gr.Row():
                    steps = gr.Slider(1, 50, value=25, step=1, label="Steps")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                transparent_rgba = gr.Checkbox(
                    label="Transparent RGBA (prepend official transparency prompt template)",
                    value=False,
                )
                negative_prompt = gr.Textbox(label="Negative prompt", value=" ", lines=1)
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", type="pil", format="png")
                used_seed = gr.Number(label="Seed used", precision=0)
                used_prompt = gr.Textbox(label="Final prompt sent to model", lines=3)

        inputs = [
            prompt,
            reference_image,
            aspect_ratio,
            steps,
            seed,
            randomize_seed,
            transparent_rgba,
            negative_prompt,
        ]
        outputs = [output_image, used_seed, used_prompt]
        run_btn.click(fn=generate, inputs=inputs, outputs=outputs)
        prompt.submit(fn=generate, inputs=inputs, outputs=outputs)
    return demo


demo = build_demo()

if __name__ == "__main__":
    # Spaces sets GRADIO_SERVER_NAME; share=True is mainly for local/Colab ad-hoc runs
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    demo.queue(max_size=4).launch(share=share)
