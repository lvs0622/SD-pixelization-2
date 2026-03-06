import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class PixelStyleTransfer:
    def __init__(self, base_model_id: str, controlnet_id: str, device: str | None = None):
        self.device = device or _device()

        print("Loading ControlNet from:", controlnet_id)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
            local_files_only=True
        )

        print("Loading SD1.5 from:", base_model_id)
        if os.path.isdir(base_model_id):
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        else:
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to(self.device)
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        
        try:
            self.pipe.vae.enable_slicing()
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
            except Exception as e:
                print(f"LoRA load failed: {e}")
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

    def _prepare_control_image(self, image: Image.Image, tile_size: int = 64):
        orig_w, orig_h = image.size
        img_np = np.array(image)
        
        new_h = (orig_h // tile_size) * tile_size
        new_w = (orig_w // tile_size) * tile_size
        
        if new_h != orig_h or new_w != orig_w:
            image = image.resize((new_w, new_h), Image.LANCZOS)
            img_np = np.array(image)
        
        small_h = max(1, new_h // tile_size)
        small_w = max(1, new_w // tile_size)
        
        small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
        control_image = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return Image.fromarray(control_image), image

    def run(self, image: Image.Image, style_name: str, custom_prompt: str, 
            negative_prompt: str, steps: int, guidance: float, 
            strength: float, seed: int, styles: dict) -> Image.Image:
        
        cfg = styles.get(style_name, {})
        lora_path = cfg.get("lora_path")
        lora_scale = cfg.get("lora_scale", 0.9)
        control_tile_size = cfg.get("control_tile_size", 64)
        base_prompt = cfg.get("prompt", "")
        neg_prompt = cfg.get("negative_prompt", "")
        
        full_prompt = (base_prompt + ", " + custom_prompt).strip(", ").strip()
        final_negative = negative_prompt.strip() if negative_prompt else neg_prompt

        self.set_style(lora_path=lora_path, lora_scale=lora_scale)
        control_image, adjusted_image = self._prepare_control_image(image, control_tile_size)

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

        result = self.pipe(
            prompt=full_prompt,
            image=adjusted_image,
            control_image=control_image,
            negative_prompt=final_negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=generator,
            controlnet_conditioning_scale=1.0
        )
        
        return result.images[0]
