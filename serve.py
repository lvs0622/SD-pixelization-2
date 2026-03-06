import os
import json
from PIL import Image
import gradio as gr
from pixel_transfer.pipeline import PixelStyleTransfer

# 加载样式配置
def load_styles(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

styles = load_styles(os.path.join(os.path.dirname(__file__), "styles", "styles.json"))

# 初始化 pipeline
base_model = "/root/autodl-tmp/SD-pixelization-2/models/sd15/AI-ModelScope/stable-diffusion-v1-5"
controlnet_id = "/root/autodl-tmp/SD-pixelization-2/models/controlnet-tile"
pipe = PixelStyleTransfer(base_model_id=base_model, controlnet_id=controlnet_id)

# 生成函数
def generate(image, style_name, custom_prompt, negative_prompt, steps, guidance, strength, seed):
    return pipe.run(
        image=image,
        style_name=style_name,
        custom_prompt=custom_prompt or "",
        negative_prompt=negative_prompt or "",
        steps=steps,
        guidance=guidance,
        strength=strength,
        seed=int(seed),
        styles=styles
    )

# Gradio UI
with gr.Blocks(title="像素风格迁移") as demo:
    gr.Markdown("像素风格迁移（Stable Diffusion 1.5 + LoRA + ControlNet Tile）")
    with gr.Row():
        with gr.Column(scale=1):
            # 移除 source="upload"，新版默认就是上传
            in_img = gr.Image(type="pil", label="原图")
            style = gr.Dropdown(choices=list(styles.keys()), value=list(styles.keys())[0], label="像素风格")
            custom_prompt = gr.Textbox(value="", label="附加提示词", placeholder="可选")
            negative = gr.Textbox(value="", label="负面提示词", placeholder="留空使用风格默认")
            steps = gr.Slider(5, 50, value=30, step=1, label="采样步数")
            guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.1, label="引导系数")
            strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="风格强度")
            seed = gr.Number(value=-1, precision=0, label="随机种子（-1随机）")
            run_btn = gr.Button("生成")
        with gr.Column(scale=1):
            out_img = gr.Image(type="pil", label="输出图像")
    run_btn.click(
        fn=generate,
        inputs=[in_img, style, custom_prompt, negative, steps, guidance, strength, seed],
        outputs=[out_img],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
