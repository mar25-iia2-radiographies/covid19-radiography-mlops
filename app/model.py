from pathlib import Path

import keras
from tensorflow.keras.applications.vgg16 import preprocess_input


# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemin vers le modèle
MODEL_PATH = BASE_DIR / "models" / "final_vgg16_fixed_splits.keras"

# Classes du modèle
CLASS_NAMES = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]


def load_covid_model():
    """
    Charge le modèle VGG16 entraîné.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

    return model


def predict_image(model, image):
    """
    Effectue une prédiction sur une image déjà prétraitée.

    image : numpy.ndarray de dimensions (1, 224, 224, 3)
            avec des valeurs sur l'échelle [0, 255].

    Retourne les probabilités des quatre classes.
    """

    predictions = model.predict(image, verbose=0)

    return predictions[0]