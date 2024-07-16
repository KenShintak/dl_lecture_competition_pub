from PIL import Image

import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor

train_image_path = '/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/train_image_paths.txt'
images_path = '/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/Images'

with open(train_image_path, 'r') as f:
    list_txt = []
    list_image_path = []
    for line in f:
      list_txt.append(line.split("/")[0])
      list_image_path.append(images_path + "/" + line.strip())

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define a custom dataset
class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function)
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True) #Define your own dataloader

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
  model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch

        images= images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

# ファインチューニングの後、モデルとプロセッサを保存

save_path = "/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/my_finetuned_clip.pt"
torch.save(model.state_dict(), save_path)