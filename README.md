# 🪙 Classification de Pièces de Monnaie avec AlexNet

**Classification d'images de pièces en utilisant le Transfer Learning avec AlexNet sur PyTorch**

> Un projet d'apprentissage profond appliquant les meilleures pratiques du transfer learning pour reconnaître automatiquement différents types de pièces. Accuracy : **~95%** sur le set de validation.

---

## 📋 Contexte & Objectif

**Problème :** Classifier automatiquement des images de pièces de monnaie en plusieurs catégories (différents pays, valeurs, années).

**Approche :** 
- Réutiliser un modèle AlexNet pré-entraîné sur ImageNet (transfer learning)
- Fine-tuner les couches supérieures du réseau sur notre dataset spécifique
- Optimiser avec stratification train/validation et augmentation de données

**Résultat :** Modèle entraîné capable de prédire la classe d'une pièce avec haute précision.

---

## 🛠️ Stack Technique

| Domaine | Outils |
|---------|--------|
| **Deep Learning** | PyTorch, torchvision |
| **Transfer Learning** | AlexNet (ImageNet) |
| **Données** | Pandas, scikit-learn (train/test split stratifié) |
| **Visualisation** | Matplotlib |
| **Accélération** | GPU CUDA (auto-détection device) |

---

## 📊 Dataset

- **Source :** DL4CV Coin Classification (Kaggle)
- **Structure :** Images PNG/JPG + CSV d'annotations (Id, Class)
- **Preprocessing :**
  - Redimensionnement 224×224 (entrée AlexNet)
  - Normalisation ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
  - Augmentation train : flips horizontaux, rotations ±10°
  - Split stratifié 80/20 train/validation

---

## 🎯 Architecture & Entraînement

AlexNet (pré-entraîné ImageNet) ↓ Freeze features (conv + pool layers) ↓ Fine-tune classifier ↓ Remplacer dernière couche : 1000 → num_classes


**Hyperparamètres :**
- **Optimizer :** SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
- **Loss :** CrossEntropyLoss
- **Scheduler :** StepLR (step_size=5, gamma=0.1)
- **Epochs :** 15
- **Batch size :** 64
- **Device :** GPU/CPU auto-détection

---

## 📈 Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy (Validation) | ~95% |
| Loss final | < 0.15 |
| Nombre de classes | 315 |
| Images d'entraînement | ~5000 |
| Images de validation | ~1250 |

**Courbes d'apprentissage :**  
- Loss converge régulièrement (pas d'overfitting significatif)
- Accuracy augmente progressivement jusqu'au plateau

---

## 🚀 Comment utiliser

### Installation

```bash
pip install torch torchvision pandas scikit-learn matplotlib tqdm
1. Préparer les données
# Les données doivent être organisées ainsi :
data/coins_split/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_2/
│       └── image3.jpg
└── val/
    ├── class_1/
    └── class_2/
2. Entraîner le modèle
python train.py --epochs 15 --batch_size 64 --lr 0.01
3. Prédire sur une nouvelle image
from PIL import Image
import torch

model = torch.load("alexnet_coins.pth")
img = Image.open("mon_image.jpg")
# Preprocess & predict
pred = model(img)
