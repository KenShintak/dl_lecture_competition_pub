import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer
from PIL import Image
import clip

# Load pre-trained CLIP model
clip_model, preprocess = clip.load("ViT-B/32", jit=False)
clip_model = CLIPModel.from_pretrained("/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/finetuned_clip")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Define a function to calculate the similarity between an image and a sentence
def calculate_similarity(image_path, sentence):
  # Process the image
  image = Image.open(image_path)
  image = preprocess(image).unsqueeze(0)  # Add batch dimension

  # Encode the image and sentence
  image_features = clip_model.get_image_features(image)
  text_features = clip_model.get_text_features(sentence)

  # Calculate cosine similarity
  similarity = F.cosine_similarity(image_features, text_features)
  return similarity.item()

# Example usage
image_path = "/content/drive/MyDrive/ColabData/dl_lecture_competition_pub/data/Images/accordion/accordion_01b.jpg"
inputs = tokenizer(["accordion", "acorn", "toothpick"], padding=True, return_tensors="pt")
text_features = clip_model.get_text_features(**inputs)

for sentence in text_features:
  similarity = calculate_similarity(image_path, sentence)
  print(f"Similarity for sentence '{sentence}': {similarity:.4f}")