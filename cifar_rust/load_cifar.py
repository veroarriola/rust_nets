#pip install datasets pillow
from datasets import load_dataset
import os

# Carga CIFAR-10 desde Hugging Face
ds = load_dataset("uoft-cs/cifar10")
clases = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for split in ["train", "test"]:
    for item in ds[split]:
        img = item["img"]
        label_name = clases[item["label"]]

        dir_path = f"cifar10_images/{split}/{label_name}"
        os.makedirs(dir_path, exist_ok=True)

        # Guardamos la imagen numerada (usando el tamaño del directorio actual)
        idx = len(os.listdir(dir_path))
        img.save(f"{dir_path}/{idx}.png")

print("¡Imágenes descargadas y organizadas!")
