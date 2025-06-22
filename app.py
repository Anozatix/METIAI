import streamlit as st
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image

# ==== Generator Definition ====
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_embed(labels)
        x = torch.cat([z, label_input], dim=1)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)

# ==== Load Trained Generator ====
device = torch.device("cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator_mnist_cgan.pth", map_location=device))
generator.eval()

# ==== Streamlit UI ====
st.set_page_config(page_title="Digit Generator", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1 {
        text-align: center;
        color: #1f1f1f;
        font-family: 'Segoe UI', sans-serif;
    }
    .digit-title {
        font-size: 18px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 20px;
    }
    button {
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("📝 Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

# --- User Input ---
digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
generate = st.button("🎨 Generate Images")

# --- Image Generation ---
if generate:
    with torch.no_grad():
        z = torch.randn(5, 100).to(device)
        labels = torch.full((5,), digit, dtype=torch.long).to(device)
        generated_imgs = generator(z, labels).cpu()

    st.markdown(f"### Generated images of digit **{digit}**")
    cols = st.columns(5)

    for i in range(5):
        img = generated_imgs[i].squeeze().numpy()
        img = (img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        img_pil = Image.fromarray(np.uint8(img * 255), mode='L').resize((100, 100))
        cols[i].image(img_pil, caption=f"Sample {i+1}", use_container_width=True)
