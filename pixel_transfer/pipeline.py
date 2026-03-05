import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import os


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _seed_generator(seed: int | None):
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=_device()).manual_seed(seed)


def _tile_down_up(img: Image.Image, tile_size: int):
    w, h = img.size
    if tile_size <= 0:
        return img
    scale_w = max(1, w // tile_size)
    scale_h = max(1, h // tile_size)
    s = min(scale_w, scale_h)
    if s <= 0:
        s = 1
    dw = max(1, w // s)
    dh = max(1, h // s)
    small = img.resize((dw, dh), resample=Image.NEAREST)
    back = small.resize((w, h), resample=Image.NEAREST)
    return back


class PixelStyleTransfer:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11f1e_sd15_tile",
        device: str | None = None,
    ):
        self.device = device or _device()
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=_dtype())
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=_dtype(),
            safety_checker=None,
        )
        self.pipe = pipe.to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.current_lora = None
        self.current_lora_scale = None

    def set_style(self, lora_path: str | None, lora_scale: float | None = None):
        if lora_path and os.path.isdir(lora_path):
            try:
                if self.current_lora and hasattr(self.pipe, "unload_lora_weights"):
                    self.pipe.unload_lora_weights()
            except Exception:
                pass
            try:
                self.pipe.load_lora_weights(lora_path)
                if hasattr(self.pipe, "set_adapters") and lora_scale is not None:
                    self.pipe.set_adapters(["default"], [float(lora_scale)])
                elif hasattr(self.pipe, "fuse_lora") and lora_scale is not None:
                    self.pipe.fuse_lora(lora_scale=float(lora_scale))
                self.current_lora = lora_path
                self.current_lora_scale = lora_scale
            except Exception:
                self.current_lora = None
                self.current_lora_scale = None
        else:
            try:
                if hasattr(self.pipe, "unload_lora_weights"):
                    self.pipe.unload_lora_weights()
            except Exception:
                pass
            self.current_lora = None
            self.current_lora_scale = None

    def run(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: int | None = None,
        control_tile_size: int = 64,
    ) -> Image.Image:
        control_img = _tile_down_up(image.convert("RGB"), control_tile_size)
        generator = _seed_generator(seed)
        out = self.pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            control_image=control_img,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            generator=generator,
        )
        return out.images[0]

