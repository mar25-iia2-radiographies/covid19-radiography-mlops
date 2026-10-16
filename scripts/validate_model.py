import numpy as np

from app.model import (
    load_covid_model,
    predict_image,
    CLASS_NAMES
)


def main():

    print("=" * 50)
    print("VALIDATION DU MODÈLE VGG16")
    print("=" * 50)

    # 1. Chargement du modèle
    print("\n1. Chargement du modèle...")

    model = load_covid_model()

    print("Modèle chargé avec succès.")

    # 2. Vérification de l'architecture
    print("\n2. Architecture du modèle")

    print("Input shape :", model.input_shape)
    print("Output shape :", model.output_shape)

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 4)

    # 3. Création d'une image artificielle
    print("\n3. Création d'une image de test")

    rng = np.random.default_rng(42)

    image = rng.integers(
        low=0,
        high=256,
        size=(1, 224, 224, 3),
        dtype=np.uint8
    ).astype(np.float32)

    print("Image shape :", image.shape)

    # 4. Prédiction
    print("\n4. Inférence")

    probabilities = predict_image(model, image)

    for label, probability in zip(CLASS_NAMES, probabilities):
        print(f"{label}: {probability:.4f}")

    # 5. Vérification des résultats
    print("\n5. Vérifications")

    assert probabilities.shape == (4,)

    assert np.all(np.isfinite(probabilities))

    assert np.all(probabilities >= 0)

    assert np.all(probabilities <= 1)

    assert np.isclose(
        np.sum(probabilities),
        1.0,
        atol=1e-5
    )

    predicted_class = CLASS_NAMES[
        np.argmax(probabilities)
    ]

    print("\nClasse prédite :", predicted_class)

    print("\nToutes les vérifications sont réussies !")


if __name__ == "__main__":
    main()