"""Prétraitement identique au notebook vgg16-clahe-f-augment-f.ipynb."""
from pathlib import Path
import cv2
import numpy as np

TARGET_SIZE = 224

def preprocess_image(image_path: str | Path, mask_path: str | Path) -> np.ndarray:
    """Retourne un tableau RGB float32 (224,224,3), pixels [0,255].

    Ne pas appliquer keras.applications.vgg16.preprocess_input ici :
    cette transformation est déjà dans le modèle Keras sauvegardé.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de lire la radiographie : {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossible de lire le masque : {mask_path}")
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    image = image * mask
    image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.astype(np.float32)


def make_batch(image: np.ndarray) -> np.ndarray:
    if image.shape != (TARGET_SIZE, TARGET_SIZE, 3):
        raise ValueError(f"Dimensions non conformes : {image.shape}")
    if image.dtype != np.float32:
        raise ValueError(f"Type non conforme : {image.dtype}")
    return np.expand_dims(image, axis=0)
