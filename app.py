import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
class DoubleConv(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return torch.sigmoid(self.final(x))

model = UNet().to(device)

model_path = "best_model.pth"

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Make sure best_model.pth is in repo.")

# Load safely
state_dict = torch.load(model_path, map_location=device)

# Fix key mismatch (if any)
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

try:
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    import traceback
    st.text("FULL ERROR:")
    st.text(traceback.format_exc())

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128,128))
    image = image / 255.0
    image = np.expand_dims(image, axis=(0,1))
    return torch.tensor(image, dtype=torch.float32)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🦠 Malaria Cell Segmentation (U-Net)")
st.write("Upload a blood smear image to segment infected cells.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    input_tensor = preprocess(img)

    with torch.no_grad():
        pred = model(input_tensor).squeeze().numpy()

    pred_mask = (pred > 0.5).astype(float)

    # Display
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image")

    with col2:
        st.image(pred_mask, caption="Predicted Mask")
