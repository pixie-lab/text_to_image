import os
import random
import uuid
import json

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "sd-community/sdxl-flash",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")


def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=30, queue=False)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    num_inference_steps: int = 30,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    pipe.to(device)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)   

    options = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "width":width,
        "height":height,
        "guidance_scale":guidance_scale,
        "num_inference_steps":num_inference_steps,
        "generator":generator,
        "use_resolution_binning":use_resolution_binning,
        "output_type":"pil",

    }
    
    images = pipe(**options).images

    image_paths = [save_image(img) for img in images]
    return image_paths, seed


#examples = [
#   "a cat eating a piece of cheese",
#  "a ROBOT riding a BLUE horse on Mars, photorealistic, 4k"
#]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown("""# SDXL Flash
        ### First Image processing takes time then images generate faster.""")
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Describe image to create",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1)
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=5,
                lines=4,
                placeholder="Enter a negative prompt",
                value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                visible=True,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=64,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=64,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=6,
                step=0.1,
                value=3.0,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=15,
                step=1,
                value=8,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)