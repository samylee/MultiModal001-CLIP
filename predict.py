import torch
from clip import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("weights/ViT-B-32.pt", device=device)

# Resize/CenterCrop/_convert_image_to_rgb/ToTensor/Normalize
image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Label probs: [[0.001089 0.00897  0.9897  ]]
print("Label probs:", probs)