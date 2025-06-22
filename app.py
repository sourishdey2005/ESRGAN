import gradio as gr
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import time
import gc
import gdown

from RRDBNet_arch import RRDBNet

# -------------------------
# Download from Google Drive if not present
# -------------------------
def ensure_model_downloaded():
    model_path = "models/RRDB_ESRGAN_x4.pth"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        file_id = "1P3Hbr51ZNsbNJIiWxrsHgl-D3I9n5ItN"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# -------------------------
# Load ESRGAN Model
# -------------------------
@torch.no_grad()
def load_model():
    ensure_model_downloaded()
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
    model_path = os.path.join("models", "RRDB_ESRGAN_x4.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    return model

model = load_model()

# -------------------------
# Utility Functions
# -------------------------
def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img_pil).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze().detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)

def fuse_images(img1, img2):
    img1 = img1.resize((384, 384), Image.LANCZOS)
    img2 = img2.resize((384, 384), Image.LANCZOS)
    return Image.blend(img1, img2, alpha=0.5)

def sharpen_image(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=1))

def upscale_to_resolution(img: Image.Image, resolution: str = "4K") -> Image.Image:
    target_size = (3840, 2160) if resolution == "4K" else (7680, 4320)
    return img.resize(target_size, Image.LANCZOS)

# -------------------------
# Inference Pipeline
# -------------------------
def esrgan_pipeline(img1, img2, resolution):
    if not img1 or not img2:
        return None, None, "Please upload two valid images."

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    fused_img = fuse_images(img1, img2)

    start = time.time()

    with torch.no_grad():
        input_tensor = preprocess(fused_img)
        sr1 = model(input_tensor)
        sr2 = model(sr1)
        sr3 = model(sr2)

    base_output = postprocess(sr3)

    gc.collect()
    torch.cuda.empty_cache()

    upscaled_img = upscale_to_resolution(base_output, resolution)
    final_img = sharpen_image(upscaled_img)

    elapsed = time.time() - start
    sharpness_score = torch.var(torch.tensor(base_output.convert("L"))).item()
    msg = f"✅ Done in {elapsed:.2f}s | Sharpness: {sharpness_score:.2f}"

    return base_output, final_img, msg

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="Triple-Pass ESRGAN Super-Resolution") as demo:
    gr.Markdown("## 🧠 Triple-Pass ESRGAN Ultra-HD Upscaler")
    gr.Markdown("Upload **two low-res images** → ESRGAN (3 passes) → Final **4K/8K** enhanced image with sharpening.")

    with gr.Row():
        with gr.Column():
            img_input1 = gr.Image(type="pil", label="Low-Res Image 1")
            img_input2 = gr.Image(type="pil", label="Low-Res Image 2")
            resolution_choice = gr.Radio(["4K", "8K"], value="4K", label="Select Output Resolution")
            run_button = gr.Button("🚀 Run ESRGAN")

        with gr.Column():
            output_esrgan = gr.Image(label="🧠 ESRGAN 3x Output")
            output_final = gr.Image(label="🏞️ Final Enhanced Output")
            result_text = gr.Textbox(label="📊 Output Log")

    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align: center; font-size: 16px;'>"
        "Made with ❤️ by <b>CodeKarma</b> as a part of <b>Bharatiya Antariksh Hackathon 2025</b>"
        "</div>",
        unsafe_allow_html=True
    )

    run_button.click(fn=esrgan_pipeline,
                     inputs=[img_input1, img_input2, resolution_choice],
                     outputs=[output_esrgan, output_final, result_text])

# -------------------------
# Launch
# -------------------------
if __name__ == "__main__":
    demo.launch()
