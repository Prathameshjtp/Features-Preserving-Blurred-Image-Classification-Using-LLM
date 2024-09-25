import argparse
import gradio as gr
from PIL import Image
import os
import torch
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

## local code
from models import instructir
from text.models import LanguageModel, LMHead

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt", local_dir="./")
hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt", local_dir="./")

CONFIG     = "configs/eval5d.yml"
LM_MODEL   = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"

# parse config file
with open(os.path.join(CONFIG), "r") as f:
    config = yaml.safe_load(f)

cfg = dict2namespace(config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks,
                            middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
model = model.to(device)
print ("IMAGE MODEL CKPT:", MODEL_NAME)
model.load_state_dict(torch.load(MODEL_NAME, map_location="cpu"), strict=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LMODEL = cfg.llm.model
language_model = LanguageModel(model=LMODEL)
lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
lm_head = lm_head.to(device)

print("LMHEAD MODEL CKPT:", LM_MODEL)
lm_head.load_state_dict(torch.load(LM_MODEL, map_location="cpu"), strict=True)

def load_img (filename, norm=True,):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:
        img = img / 255.
        img = img.astype(np.float32)
    return img

def process_img (image, prompt):
    img = np.array(image)
    img = img / 255.
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    lm_embd = language_model(prompt)
    lm_embd = lm_embd.to(device)

    with torch.no_grad():
        text_embd, deg_pred = lm_head (lm_embd)
        x_hat = model(y, text_embd)

    restored_img = x_hat.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    restored_img = np.clip(restored_img, 0. , 1.)

    restored_img = (restored_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return Image.fromarray(restored_img)

css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
"""

demo = gr.Interface(
    fn=process_img,
    inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Text(label="Prompt")
    ],
    outputs=[gr.Image(type="pil", label="Ouput")],
    css=css,
)

if __name__ == "__main__":
    demo.launch(True)
