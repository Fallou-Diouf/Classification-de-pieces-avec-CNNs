# 🪙 Classification de pièces de monnaie avec AlexNet

> Classification automatique de pièces de monnaie par **Deep Learning** et **Transfer Learning** avec **PyTorch** et **AlexNet** pré-entraîné sur ImageNet.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-green)
![AlexNet](https://img.shields.io/badge/Model-AlexNet-orange)

---

## Présentation

Ce projet met en œuvre un système de **classification d'images de pièces de monnaie** à l'aide d'un réseau de neurones convolutif (**CNN**) basé sur **AlexNet**.

L'objectif est de comparer une approche basée sur le **Deep Learning** avec une approche classique de vision par ordinateur utilisée précédemment pour la reconnaissance de pièces.

Les méthodes classiques telles que la **détection de contours de Canny** ou la **transformée de Hough** permettent d'extraire des caractéristiques géométriques, mais deviennent limitées lorsque les images présentent des variations d'éclairage, d'orientation, de bruit ou d'apparence.

Avec un CNN, les caractéristiques visuelles sont directement apprises à partir des images.

### Objectif

Construire un modèle capable de :

* reconnaître automatiquement le type d'une pièce ;
* apprendre des représentations visuelles à partir des images ;
* exploiter un modèle pré-entraîné sur ImageNet ;
* évaluer les performances sur un jeu de validation ;
* analyser les limites du modèle et les pistes d'amélioration.

---

## Résultats

Le modèle obtient une **accuracy d'environ 95 % sur le jeu de validation**.

| Métrique                  |                          Résultat |
| ------------------------- | --------------------------------: |
| Accuracy validation    |                         **~95 %** |
| Nombre de classes     |                           **315** |
| Images d'entraînement |                        **~5 000** |
| Images de validation   |                        **~1 250** |
| Modèle                 | **AlexNet pré-entraîné ImageNet** |

> Les résultats peuvent varier légèrement selon la version du dataset, le split train/validation et les paramètres d'entraînement.

---

## Dataset

Le projet utilise le dataset **DL4CV Coin Classification** provenant d'un challenge Kaggle.

Le dataset contient des images de pièces provenant de différentes devises et différents pays, avec une classe associée à chaque image.

### Structure initiale

Les données sont fournies sous la forme d'un fichier CSV contenant notamment :

```text
Id, Class
```

ainsi qu'un répertoire contenant les images.

### Préparation du dataset

Le dataset est automatiquement réorganisé pour être compatible avec `torchvision.datasets.ImageFolder`.

```text
data/
└── coins_split/
    ├── train/
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   ├── class_2/
    │   │   └── image3.jpg
    │   └── ...
    │
    └── val/
        ├── class_1/
        ├── class_2/
        └── ...
```

Le split est réalisé avec une **stratification des classes** afin de conserver une distribution similaire entre les ensembles d'entraînement et de validation.

```python
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Class_clean"],
    random_state=42
)
```

---

## Pipeline

Le pipeline complet est le suivant :

```text
Dataset Kaggle
      │
      ▼
Chargement du CSV
      │
      ▼
Vérification des images
      │
      ▼
Nettoyage des labels
      │
      ▼
Split stratifié 80 / 20
      │
      ▼
Data Augmentation
      │
      ▼
Redimensionnement 224 × 224
      │
      ▼
Normalisation ImageNet
      │
      ▼
AlexNet pré-entraîné
      │
      ▼
Adaptation du classifieur
      │
      ▼
Entraînement avec PyTorch
      │
      ▼
Évaluation
      │
      ▼
~95 % Accuracy
```

---

## Architecture du modèle

Le modèle utilisé est **AlexNet**, pré-entraîné sur ImageNet.

AlexNet est composé principalement de :

```text
Input Image
    │
    ▼
Convolutional Layers
    │
    ▼
ReLU + Pooling
    │
    ▼
Feature Extraction
    │
    ▼
Fully Connected Layers
    │
    ▼
Classification
```

Le classifieur original d'AlexNet produit des prédictions pour **1000 classes ImageNet**.

Il est donc adapté au dataset de pièces :

```python
num_classes = len(train_dataset.classes)

model.classifier[6] = nn.Linear(
    4096,
    num_classes
)
```

Le modèle devient ainsi capable de prédire les **315 classes de pièces** du dataset.

---

## Transfer Learning

Le projet utilise le **Transfer Learning** afin de réutiliser les représentations visuelles apprises par AlexNet sur ImageNet.

Les premières couches d'un CNN apprennent généralement des caractéristiques génériques telles que :

* les contours ;
* les textures ;
* les formes ;
* les motifs visuels.

Ces représentations peuvent ensuite être adaptées à une nouvelle tâche de classification.

---

## Prétraitement des images

AlexNet attend des images de taille :

```text
224 × 224 pixels
```

### Augmentation des données

Pour l'entraînement :

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
```

Les transformations permettent d'améliorer la robustesse du modèle face aux variations des images.

### Validation

Pour la validation, aucune augmentation aléatoire n'est appliquée :

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
```

---

## Configuration de l'entraînement

| Paramètre        | Valeur           |
| ---------------- | ---------------- |
| Modèle           | AlexNet          |
| Pré-entraînement | ImageNet         |
| Optimiseur       | SGD              |
| Learning rate    | `0.01`           |
| Momentum         | `0.9`            |
| Weight decay     | `1e-4`           |
| Loss             | CrossEntropyLoss |
| Scheduler        | StepLR           |
| Step size        | `5`              |
| Gamma            | `0.1`            |
| Epochs           | `15`             |
| Batch size       | `64`             |
| Input size       | `224 × 224`      |
| Device           | CUDA / CPU       |

Le device est automatiquement sélectionné :

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## Sauvegarde du modèle

Le modèle entraîné est sauvegardé sous forme de `state_dict` :

```python
torch.save(
    model.state_dict(),
    "alexnet_coins.pth"
)

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Fallou-Diouf/nom-du-repo.git
cd nom-du-repo
```

### 2. Installer les dépendances

```bash
pip install torch torchvision pandas scikit-learn matplotlib tqdm
```

---

## ▶️ Entraînement

Après avoir préparé le dataset :

```bash
python train.py
```

Les principaux paramètres peuvent être configurés directement dans le script :

```python
num_epochs = 15
batch_size = 64
learning_rate = 0.01
```

---

## 🔮 Prédiction sur une nouvelle image

Une image peut ensuite être prétraitée avec les mêmes transformations que celles utilisées lors de la validation.

```python
from PIL import Image
import torch

image = Image.open("coin.jpg").convert("RGB")

image = val_transform(image)
image = image.unsqueeze(0)
image = image.to(device)

model.eval()

with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1)

print("Classe prédite :", prediction.item())
```

---

## 🧪 Améliorations possibles

Plusieurs pistes peuvent être explorées pour améliorer le projet.

### 🔹 1. Comparer Feature Extraction et Fine-Tuning

Comparer :

```text
AlexNet + Frozen Features
```

avec :

```text
AlexNet + Fine-Tuning
```

afin de mesurer l'impact de l'adaptation des couches convolutionnelles.

### 🔹 2. Tester d'autres architectures

Comparer AlexNet avec des architectures plus modernes :

* ResNet18
* ResNet50
* EfficientNet
* MobileNet
* Vision Transformer (ViT)

### 🔹 3. Améliorer la Data Augmentation

Tester :

* RandomCrop ;
* ColorJitter ;
* RandomResizedCrop ;
* rotations plus importantes ;
* transformations photométriques.

### 🔹 4. Évaluer avec davantage de métriques

Au-delà de l'accuracy :

* Precision ;
* Recall ;
* F1-score ;
* matrice de confusion ;
* top-5 accuracy.

### 🔹 5. Analyse des erreurs

Identifier les classes régulièrement confondues par le modèle.

Cela permettrait notamment d'étudier si certaines pièces ont une apparence très similaire.

---

## 📚 Ce que ce projet permet de mettre en pratique

Ce projet constitue une première mise en pratique de plusieurs concepts importants en **Computer Vision** et **Deep Learning** :

* CNN ;
* Transfer Learning ;
* Image Classification ;
* Data Augmentation ;
* Image Normalization ;
* Stratified Train/Validation Split ;
* PyTorch `Dataset` et `DataLoader` ;
* GPU avec CUDA ;
* Optimisation par SGD ;
* Learning Rate Scheduling ;
* Évaluation d'un modèle ;
* Sauvegarde et chargement des poids.

---

## 🧠 Compétences développées

**Computer Vision**

* Classification d'images
* Prétraitement d'images
* Data Augmentation
* Analyse des performances

**Deep Learning**

* CNN
* Transfer Learning
* Fine-Tuning
* Fonction de perte
* Optimisation
* Learning Rate Scheduling

**PyTorch**

* `Dataset`
* `DataLoader`
* `torchvision`
* Entraînement GPU
* Évaluation
* Sauvegarde des modèles

---

## 📖 Références

**Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).**

*ImageNet Classification with Deep Convolutional Neural Networks.*

Advances in Neural Information Processing Systems (NeurIPS).

---

## 👨‍💻 Auteur

**Fallou Diouf**

🎓 MSc — Vision & Machine Intelligence
🔬 Computer Vision • Deep Learning • Information Retrieval

Ce projet a été réalisé dans le cadre de la formation en **Vision par Ordinateur et Réseaux de Neurones**.
