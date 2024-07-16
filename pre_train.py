import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def cosine_similarity(a, b):
  #a, b: 128x512
  #出力: 128x128

  b_t = b.transpose(0,1)
  c = torch.matmul(a,b_t)

  a_norm = torch.norm(a, p=2, dim=1)
  b_norm = torch.norm(b, p=2, dim=1)

  a_norm = a_norm.reshape(-1,1)
  b_norm = b_norm.unsqueeze(0)

  d = torch.matmul(a_norm, b_norm)
  return c / d

def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

def clip_loss(clip_z, z, temperature=0.5):
  #clip_z, zのサイズ：(128or64, 512)
  B = z.size(0)
  cosine_sim_matrix = cosine_similarity(clip_z, z)
  cosine_sim_matrix = torch.exp(cosine_sim_matrix/temperature) #exp(cos類似度/temperature)で成る行列
  item1_a = torch.diagonal(cosine_sim_matrix)
  item2_a = torch.diagonal(cosine_sim_matrix)
  item1_b = torch.sum(cosine_sim_matrix, dim=0)
  item2_b = torch.sum(cosine_sim_matrix, dim=1)
  item = -torch.sum(torch.log(item1_a/item1_b) + torch_log(item2_a/item2_b)) / B
  return item

def SE_loss(clip_z, z):
  a = clip_z - z
  a = torch.pow(a, 2)
  a = torch.sum(a)
  return a / (z.size(0) * z.size(1))

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
  set_seed(args.seed)
  loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
  train_set = ThingsMEGDataset("train", args.data_dir)
  train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
  # ------------------
  #       Model
  # ------------------
  model = BasicConvClassifier(
      train_set.num_classes, train_set.seq_len, train_set.num_channels, flag=1
  ).to(args.device)
  # ------------------
  #Load pre-trained CLIP model
  # ------------------
  clip_model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
  save_path = "/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/my_finetuned_clip.pt"
  clip_model.load_state_dict(torch.load(save_path))
  
  # ------------------
  #     Optimizer
  # ------------------
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-04)
  epochs = 8
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    total_loss = 0
    flag = 0
    model.train()
    
    for X, y, subject_idxs, image_paths in tqdm(train_loader, desc="Train"):
        clip_z = torch.empty(X.size(0), 512)
        for i, image_path in enumerate(image_paths):
          img = Image.open(image_path)
          image = preprocess(img).unsqueeze(0).to(args.device)
          with torch.no_grad():
            image_features = clip_model.encode_image(image)
          clip_z[i] = image_features
        X, y = X.to(args.device), y.to(args.device)
        z = model(X)
        clip_z = clip_z.to(args.device)
        l_clip = clip_loss(clip_z, z, temperature=0.5)
        l_mse = SE_loss(clip_z, z)
        loss = (l_clip + l_mse) / 2 #lambda=0.5
        #print("l_clip: ", l_clip.item(), "l_mse: ", l_mse.item(), "loss: ", loss.item())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        flag += 1
    total_loss /= flag
    print(f"total loss: {total_loss}")
    torch.save(model.state_dict(), "/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/model/after_{}_epoch_pretrain_model.pt".format(epoch))
if __name__ == "__main__":
    run()