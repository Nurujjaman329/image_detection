<div align="center">

# 🔍 Visual Image Similarity Search Engine
### Deep Feature Extraction (ResNet50) · Nearest-Neighbor Search (FAISS) · PyTorch

A reverse image search CLI tool that indexes local image collections and retrieves visually similar images using deep embeddings and cosine similarity.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0466C8?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![CUDA](https://img.shields.io/badge/CUDA-Hardware%20Accel-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

</div>

---

### ⚡ Key Features

* **Deep Representation:** Uses ImageNet-pretrained **ResNet50** (headless) to compute 2048-dimensional dense visual feature embeddings.
* **Vector Indexing:** Uses **FAISS `IndexFlatIP`** with $L_2$-normalized vectors, producing exact cosine similarity matches.
* **Flexible Queries:** Accepts either a local image file path or any direct HTTP/HTTPS web image URL.
* **Hardware Acceleration:** Automatically routes tensor operations to NVIDIA CUDA GPUs when available, with clean CPU fallback.

---

### 📁 Project Structure

```text
image_detection/
├── images/             # Target gallery images to index (.jpg, .jpeg, .png, .jfif)
├── query/              # Optional sample query images
├── image_search.py     # Main engine: pipeline, index builder, and CLI
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
