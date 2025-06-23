import io
import os
import argparse
import faiss
import numpy as np
import requests
import torch
from PIL import Image
from torchvision import models, transforms

# Fix for OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper to estimate memory usage of a PIL Image in bytes
def get_image_memory_size(img: Image.Image) -> int:
    width, height = img.size
    mode_to_bpp = {
        '1': 1/8,    # 1 bit per pixel
        'L': 1,      # grayscale
        'P': 1,      # paletted
        'RGB': 3,    # RGB
        'RGBA': 4,   # RGBA
    }
    bpp = mode_to_bpp.get(img.mode, 3)  # default to 3 if unknown
    return int(width * height * bpp)

# Feature extractor class
class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract(self, img: Image.Image) -> np.ndarray:
        img_t = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = self.model(img_t).squeeze().cpu().numpy()
        feat /= np.linalg.norm(feat)  # Normalize
        return feat.astype('float32')

# Load image from local path
def load_image_from_path(path):
    try:
        img = Image.open(path).convert('RGB')
        mem_bytes = get_image_memory_size(img)
        print(f"Loaded {path} ({img.size[0]}x{img.size[1]}) approx {mem_bytes / (1024**2):.2f} MB in RAM")
        return img
    except Exception as e:
        raise ValueError(f"Error loading image from path: {path} - {e}")

# Load image from URL
def load_image_from_url(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        mem_bytes = get_image_memory_size(img)
        print(f"Downloaded {url} ({img.size[0]}x{img.size[1]}) approx {mem_bytes / (1024**2):.2f} MB in RAM")
        return img
    except Exception as e:
        raise ValueError(f"Error loading image from URL: {url} - {e}")

# Image Search Engine
class ImageSearchEngine:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.feat_ext = FeatureExtractor()
        self.image_paths = []
        self.features = []

    def build_index(self):
        print("Loading images and extracting features...")
        for fname in os.listdir(self.image_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
                full_path = os.path.join(self.image_folder, fname)
                try:
                    img = load_image_from_path(full_path)
                    feat = self.feat_ext.extract(img)
                    self.features.append(feat)
                    self.image_paths.append(full_path)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
        self.features = np.array(self.features)
        print(f"Extracted features from {len(self.features)} images.")

        if len(self.features) == 0:
            raise ValueError("No valid images found to index.")

        dim = self.features.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (features are normalized)
        self.index.add(self.features)
        print("FAISS index built.")

    def search(self, query_img: Image.Image, top_k=3):
        query_feat = self.feat_ext.extract(query_img).reshape(1, -1)
        D, I = self.index.search(query_feat, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append({
                'image_path': self.image_paths[idx],
                'similarity': float(dist)
            })
        return results

# Entry point
def main():
    parser = argparse.ArgumentParser(description='Image Search Engine')
    parser.add_argument('--image_folder', type=str, default='images', help='Folder with images to index')
    parser.add_argument('--query', type=str, required=True, help='Path or URL of query image')
    args = parser.parse_args()

    engine = ImageSearchEngine(args.image_folder)
    engine.build_index()

    # Load query image
    if args.query.startswith('http'):
        print(f"Downloading query image from URL: {args.query}")
        query_img = load_image_from_url(args.query)
    else:
        print(f"Loading local query image: {args.query}")
        query_img = load_image_from_path(args.query)

    results = engine.search(query_img, top_k=3)

    print("\nTop similar images:")
    for i, r in enumerate(results, 1):
        print(f"{i}. Image: {r['image_path']} | Similarity: {r['similarity']:.4f}")

if __name__ == '__main__':
    main()
