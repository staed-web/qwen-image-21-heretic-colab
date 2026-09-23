"""
Gradio app: Qwen-Image-2.1 + Heretic text encoder.
For Hugging Face Spaces (ZeroGPU) or local GPU.
"""
import gc
import os
import random
import types

import gradio as gr
import torch
from PIL import Image

try:
    import spaces
except ImportError:
    spaces = None

BASE_MODEL = "Qwen/Qwen-Image-2.1"
HERETIC_TE = "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic"
DTYPE = torch.bfloat16

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

pipe = None


def patch_rope_device_sync(model):
    """Fix inv_freq (CPU) vs position_ids (CUDA) under accelerate CPU offload."""
    patched = 0
    for module in model.modules():
        if not hasattr(module, "inv_freq"):
            continue
        if getattr(module, "_heretic_rope_patch", False):
            continue
        orig_forward = module.forward

        def make_forward(orig):
            def forward(self, *args, **kwargs):
                position_ids = kwargs.get("position_ids", None)
                if position_ids is None and len(args) >= 2:
                    position_ids = args[1]
                x = args[0] if args else kwargs.get("x", None)
                device = None
                if position_ids is not None and hasattr(position_ids, "device"):
                    device = position_ids.device
                elif x is not None and hasattr(x, "device"):
                    device = x.device
                if device is not None and self.inv_freq.device != device:
                    self.inv_freq.data = self.inv_freq.data.to(device=device, dtype=self.inv_freq.dtype)
                return orig(*args, **kwargs)

            return forward

        module.forward = types.MethodType(make_forward(orig_forward), module)
        module._heretic_rope_patch = True
        patched += 1
    print(f"Patched {patched} RoPE module(s) for offload device sync")


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import QwenImage21Pipeline

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText as Qwen3VLForConditionalGeneration

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        HERETIC_TE, torch_dtype=DTYPE
    )
    patch_rope_device_sync(text_encoder)

    pipe = QwenImage21Pipeline.from_pretrained(
        BASE_MODEL, text_encoder=text_encoder, torch_dtype=DTYPE
    )
    patch_rope_device_sync(pipe.text_encoder)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    return pipe


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
):
    load_pipeline()

    if not prompt or not str(prompt).strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed or seed is None:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    final_prompt = apply_transparency_template(str(prompt), bool(transparent_rgba))
    width, height = ASPECT_RATIOS.get(aspect_ratio, (768, 768))
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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with torch.inference_mode():
            out = pipe(**kwargs).images[0]
    except torch.OutOfMemoryError as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "GPU out of memory. Use 768x768 or smaller and fewer steps. " + str(e)
        ) from e

    return out, seed, final_prompt


if spaces is not None:
    generate = spaces.GPU(duration=150)(_generate_impl)
else:
    generate = _generate_impl


with gr.Blocks(title="Qwen-Image-2.1 + Heretic TE") as demo:
    gr.Markdown(
        """
        # Qwen-Image-2.1 + Heretic Text Encoder
        Base: `Qwen/Qwen-Image-2.1` | TE: `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic`
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=4)
            reference_image = gr.Image(
                label="Optional reference image (edit / I2I)", type="pil", image_mode="RGBA"
            )
            aspect_ratio = gr.Dropdown(
                label="Aspect ratio",
                choices=list(ASPECT_RATIOS.keys()),
                value="1:1 (768x768)",
            )
            with gr.Row():
                steps = gr.Slider(1, 50, value=25, step=1, label="Steps")
                seed = gr.Number(value=42, precision=0, label="Seed")
            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
            transparent_rgba = gr.Checkbox(label="Transparent RGBA template", value=False)
            negative_prompt = gr.Textbox(label="Negative prompt", value=" ", lines=1)
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", type="pil", format="png")
            used_seed = gr.Number(label="Seed used", precision=0)
            used_prompt = gr.Textbox(label="Final prompt sent to model", lines=3)

    inputs = [
        prompt, reference_image, aspect_ratio, steps, seed,
        randomize_seed, transparent_rgba, negative_prompt,
    ]
    outputs = [output_image, used_seed, used_prompt]
    run_btn.click(fn=generate, inputs=inputs, outputs=outputs)
    prompt.submit(fn=generate, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.queue(max_size=4).launch(share=os.getenv("GRADIO_SHARE", "0") == "1")
