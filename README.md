# 🪙 Classification de Pièces de Monnaie avec AlexNet

**Classification d'images de pièces en utilisant le Transfer Learning avec AlexNet sur PyTorch**

> Un projet d'apprentissage profond appliquant les meilleures pratiques du transfer learning pour reconnaître automatiquement différents types de pièces. Accuracy : **~95%** sur le set de validation.

---

## 📋 Contexte & Objectif

Dans le cadre du module IFLBE055 ? Réseaux de neurones pour la vision par ordina-
teur, les séances 3 et 4 sont consacrées à l'introduction des réseaux de neurones convolutifs
(CNN) et à leur application à des problèmes de classi?cation d'images.
Lors d'un précédent travail pratique, une première approche de classi?cation de pièces
en euros avait été réalisée en utilisant des méthodes classiques de vision par ordinateur,
notamment des techniques de détection de contours telles que l'algorithme de Canny
ainsi que la transformée de Hough. Bien que ces méthodes permettent d'extraire certaines
caractéristiques géométriques des objets, elles restent limitées face à la variabilité des
images (éclairage, orientation, bruit, diversité des pièces).
Dans ce nouveau travail pratique, une approche basée sur l'apprentissage profond est
adoptée. L'objectif est de concevoir un système de classi?cation automatique capable
d'identi?er le type d'une pièce à partir d'une image, en s'appuyant sur un réseau de
neurones convolutif (CNN). Contrairement aux méthodes classiques, les CNN permettent
d'apprendre automatiquement des représentations pertinentes directement à partir des
données, sans nécessiter d'extraction manuelle de caractéristiques.
Les données utilisées proviennent du challenge DL4CV Coin Classi?cation proposé sur
Kaggle. Ce jeu de données contient des images de pièces issues de di?érentes devises et
pays, associées à des étiquettes décrivant leur valeur et leur origine.
L'objectif principal de ce travail est donc de mettre en ÷uvre un modèle de type Alex-
Net a?n de classi?er ces images, d'évaluer ses performances, et de comparer les résultats
obtenus avec ceux du leaderboard du challenge. Une analyse critique des résultats sera
également menée, accompagnée de propositions d'amélioration.
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

- **Source :** Les données seront issues du challenge DL4CV Coin classification | Kaggle 
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

## Comment utiliser

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

---

## Références

Alex Krizhevsky, Ilya Sutskever, Geo?rey E. Hinton.
ImageNet Classi?cation with Deep Convolutional Neural Networks.
Advances in Neural Information Processing Systems (NeurIPS), 2012.
