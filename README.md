# Qwen-Image-2.1 + Heretic Text Encoder (Colab / Gradio)

One-click **Google Colab** notebook and a standalone **Gradio** app that runs [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) with the community **Heretic (abliterated)** text encoder:

- Transformers bf16 TE: [pottokao/Qwen-Image-2.1-Text-Encoder-Heretic](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic)
- **Not** the ComfyUI-only GGUF: [pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF)

## Open in Colab

1. Open the notebook (Open-in-Colab badge / URL from this repo once published).
2. **Runtime → Change runtime type → GPU** (T4 is OK with CPU offload).
3. **Runtime → Run all**.
4. Wait for model downloads (**~25–35 GB+** on first run; can take 15–40+ minutes).
5. In the last cell output, copy the line:

   `Running on public URL: https://xxxx.gradio.live`

   That is your **temporary** Gradio share link (~72 hours). It is **not** a permanent hosted demo.

### Badge (fill in after publish)

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/Qwen_Image_2_1_Heretic_Gradio.ipynb)
```

## Files

| File | Purpose |
|------|---------|
| `Qwen_Image_2_1_Heretic_Gradio.ipynb` | Colab notebook: install → load Heretic TE + pipeline → Gradio `share=True` |
| `app.py` | Standalone Gradio app (HF Spaces / ZeroGPU style) |
| `requirements.txt` | Spaces / local dependencies |

## Hugging Face Spaces

1. Create a new Space (Gradio SDK, GPU or ZeroGPU).
2. Upload `app.py` + `requirements.txt` (+ this README).
3. ZeroGPU: `app.py` wraps generate with `@spaces.GPU` when `spaces` is importable.
4. Expect a long cold start (large model download). Free Spaces may OOM or time out; Colab with T4 + offload is the more reliable free path.

There is **no permanent live Gradio URL** shipped with this repo unless you deploy it yourself.

## How the Heretic TE is loaded

```python
import torch
from transformers import Qwen3VLForConditionalGeneration  # fallbacks in notebook/app
from diffusers import QwenImage21Pipeline

text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic",
    torch_dtype=torch.bfloat16,
)
pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # needed for Colab free T4 (~15 GB)
```

## UI features

- Prompt + optional **reference image** (edit / I2I via `image=` on the pipeline)
- Aspect ratios tuned for Colab (**1024 default**, 768, 1280×720, etc.) — **not** official 2048×2048
- Steps (default **25**), seed / randomize
- Optional **Transparent RGBA** checkbox that wraps the prompt with the official template:

  `This is an RGBA image with transparency. {prompt}. The image has alpha channel and the background is transparent.`

## Caveats (honesty)

| Caveat | Reality |
|--------|---------|
| VRAM | Free Colab T4 needs **CPU offload**; 2048px often OOMs — stay at 1024 or below |
| Download size | Base model + Heretic TE ≈ **30GB+** first run |
| Gradio share | Temporary (~72h); disappears when the Colab runtime dies |
| Free hosting | Colab disconnects; HF free Spaces may be too small / slow for this stack |
| GGUF | ComfyUI-only — do not use those files in this notebook |
| License | Base model: Qwen Research License. Heretic TE: Apache-2.0 (community derivative). Not affiliated with Alibaba/Qwen. |

## Official free demos (stock encoder, not Heretic)

- Blog: https://qwen.ai/blog?id=qwen-image-2.1
- Mainland China free UI: https://wuli.art/explore

## Install (local)

```bash
pip install "torch>=2.4.0" "transformers>=5.17" accelerate pillow gradio spaces
pip install git+https://github.com/huggingface/diffusers
python app.py
# or: GRADIO_SHARE=1 python app.py
```
