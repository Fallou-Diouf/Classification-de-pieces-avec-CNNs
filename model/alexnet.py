# --------------------------------------------------------------
# Projet : Classification de pièces avec AlexNet
# Dataset : DL4CV Coin Classification (Kaggle)
# --------------------------------------------------------------

# 1️ Importer les bibliothèques nécessaires
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# --------------------------------------------------------------
# 2️ Télécharger et préparer les chemins

!gdown --id 1e5jOTaVKqeAoHLi_UaOI1OrDievEDUHq
zip_path = "/content/dl4cv-coin-classification.zip"
extract_path = "/content/data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction terminée ")

base_dir = "/content/data/kaggle"              # dossier Kaggle
train_csv = os.path.join(base_dir, "train.csv")
train_img_dir = os.path.join(base_dir, "train")

output_dir = "/content/data/coins_split"       # dossier de sortie
train_out = os.path.join(output_dir, "train")
val_out = os.path.join(output_dir, "val")

# Créer les dossiers de sortie
os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)

# --------------------------------------------------------------
# 3️ Charger le CSV
df = pd.read_csv(train_csv)

# Nettoyer les noms de classes pour éviter problèmes de dossier
def clean_label(label):
    return label.replace(",", "_").replace(" ", "_")

df['Class_clean'] = df['Class'].apply(clean_label)

# --------------------------------------------------------------
# 4️ Filtrer les IDs dont l'image existe réellement
available_files = [f.split(".")[0] for f in os.listdir(train_img_dir)]
df = df[df['Id'].astype(str).isin(available_files)]

# Split stratifié train/val
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['Class_clean'], random_state=42
)

# --------------------------------------------------------------
# 5️ Fonction pour trouver le chemin complet de l'image
def find_image_path(img_id):
    # Vérifie plusieurs extensions possibles
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
        path = os.path.join(train_img_dir, f"{img_id}{ext}")
        if os.path.exists(path):
            return path
    return None

# --------------------------------------------------------------
# 6️ Fonction pour copier les images dans les bons dossiers
def copy_images(df, out_dir):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["Id"]
        label = row["Class_clean"]

        src = find_image_path(img_id)
        if src is None:
            continue  # Ignore si image manquante

        dst_dir = os.path.join(out_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src))

        if not os.path.exists(dst_path):
            shutil.copy(src, dst_path)

# Copier les images train/val
copy_images(train_df, train_out)
copy_images(val_df, val_out)

print(f"Nombre de classes : {df['Class_clean'].nunique()}")
print(f"Images train : {len(train_df)}")
print(f"Images val : {len(val_df)}")
print("Dataset réorganisé avec succès !")

# --------------------------------------------------------------
# 7️ Préparer les DataLoaders PyTorch

# Transformations pour AlexNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # flip horizontal aléatoire
    transforms.RandomRotation(10),      # rotation ±10°
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_out, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_out, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("Exemples de classes :", train_dataset.classes[:10])
print("Nombre images train:", len(train_dataset), "val:", len(val_dataset))

# --------------------------------------------------------------
# 8️ Charger AlexNet pré-entraîné et adapter la dernière couche
# Choisir automatiquement le device (GPU si disponible, sinon CPU)
# Le GPU permet d'accélérer fortement l'entraînement du réseau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# Charger le modèle AlexNet pré-entraîné sur ImageNet (transfer learning)
# Cela permet de réutiliser des features déjà apprises (bords, textures, formes)
model = models.alexnet(pretrained=True)

# Adapter la dernière couche du réseau au nombre de classes du dataset
# AlexNet original : 1000 classes → ici : num_classes (ex: 315)
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)

# Envoyer le modèle sur le device (GPU ou CPU)
model = model.to(device)

# --------------------------------------------------------------
# 9️ Définir la loss et l’optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization pour limiter l'overfitting
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# --------------------------------------------------------------
#10 Boucle d’entraînement simple
num_epochs = 15  
loss_list = []
accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    # Moyenne du loss
    epoch_loss = running_loss / len(train_loader)
    loss_list.append(epoch_loss)

    # Accuracy
    epoch_acc = 100 * correct / total
    accuracy_list.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc:.2f}%")

plt.figure(figsize=(10,4))

# Loss
plt.subplot(1,2,1)
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss curve")

# Accuracy
plt.subplot(1,2,2)
plt.plot(accuracy_list)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy curve")

plt.show()

# --------------------------------------------------------------
# 1️1 Évaluation sur le set de validation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print("Validation Accuracy:", 100*correct/total)

# --------------------------------------------------------------
# 1️2 Sauvegarder le modèle
torch.save(model.state_dict(), "/content/alexnet_coins.pth")
print("Modèle sauvegardé sous alexnet_coins.pth")