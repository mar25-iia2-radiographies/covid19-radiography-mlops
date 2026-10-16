"""Exécuter : python -m scripts.validate_real_image CHEMIN_IMAGE CHEMIN_MASQUE"""
import argparse
import numpy as np
from app.model import CLASS_NAMES, load_covid_model, predict_image
from app.preprocessing import preprocess_image, make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Chemin vers une radiographie du jeu de test')
    parser.add_argument('mask', help='Chemin vers son masque associé')
    args = parser.parse_args()
    image = preprocess_image(args.image, args.mask)
    batch = make_batch(image)
    model = load_covid_model()
    probabilities = np.asarray(predict_image(model, batch))
    print('Input:', batch.shape, batch.dtype)
    for label, proba in zip(CLASS_NAMES, probabilities):
        print(f'{label}: {proba:.6f}')
    print('Classe prédite:', CLASS_NAMES[int(np.argmax(probabilities))])
    print('Somme:', probabilities.sum())


if __name__ == '__main__':
    main()
