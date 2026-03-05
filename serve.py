import os
import json
from typing import Tuple
from PIL import Image
import gradio as gr
from pixel_transfer.pipeline import PixelStyleTransfer


def load_styles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline():
    base_model = os.getenv("BASE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    controlnet_id = os.getenv("CONTROLNET_ID", "lllyasviel/control_v11f1e_sd15_tile")
    return PixelStyleTransfer(base_model_id=base_model, controlnet_id=controlnet_id)


styles = load_styles(os.path.join(os.path.dirname(__file__), "styles", "styles.json"))
pipe = build_pipeline()


def generate(
    image: Image.Image,
    style_name: str,
    custom_prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
) -> Tuple[Image.Image]:
    cfg = styles.get(style_name, {})
    lora_path = cfg.get("lora_path")
    lora_scale = cfg.get("lora_scale", 0.9)
    control_tile_size = cfg.get("control_tile_size", 64)
    base_prompt = cfg.get("prompt", "")
    neg_prompt = cfg.get("negative_prompt", "")
    full_prompt = (base_prompt + ", " + custom_prompt).strip(", ").strip()
    final_negative = negative_prompt.strip() if negative_prompt else neg_prompt
    pipe.set_style(lora_path=lora_path, lora_scale=lora_scale)
    img = pipe.run(
        image=image,
        prompt=full_prompt,
        negative_prompt=final_negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        seed=seed if seed >= 0 else None,
        control_tile_size=control_tile_size,
    )
    return (img,)


with gr.Blocks(title="像素风格迁移") as demo:
    gr.Markdown("像素风格迁移（Stable Diffusion 1.5 + LoRA + ControlNet Tile）")
    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(type="pil", label="原图", sources=["upload", "clipboard"])
            style = gr.Dropdown(choices=list(styles.keys()), value=list(styles.keys())[0], label="像素风格")
            custom_prompt = gr.Textbox(value="", label="附加提示词", placeholder="可选")
            negative = gr.Textbox(value="", label="负面提示词", placeholder="留空使用风格默认")
            steps = gr.Slider(5, 50, value=30, step=1, label="采样步数")
            guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.1, label="引导系数")
            strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="风格强度")
            seed = gr.Number(value=-1, precision=0, label="随机种子（-1随机）")
            run_btn = gr.Button("生成")
        with gr.Column(scale=1):
            out_img = gr.Image(type="pil", label="输出图像").style(height=512)
            gr.Examples(
                examples=[],
                inputs=[in_img, style, custom_prompt, negative, steps, guidance, strength, seed],
            )
    run_btn.click(
        fn=generate,
        inputs=[in_img, style, custom_prompt, negative, steps, guidance, strength, seed],
        outputs=[out_img],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
