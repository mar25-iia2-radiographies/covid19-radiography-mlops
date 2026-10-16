"""Comparer la fonction de preprocessing du notebook VGG16 à celle du projet MLOps.

Depuis la racine du dépôt :
python -m scripts.compare_notebook_mlops data/samples/covid_test.png data/samples/covid_mask.png
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.preprocessing import make_batch, preprocess_image
from app.model import CLASS_NAMES, load_covid_model, predict_image

# Transcription de preprocess_image_for_vgg16_numpy, cellule 12 du notebook
# vgg16-clahe-f-augment-f.ipynb, avec TARGET_SIZE=224, APPLY_MASK=True,
# APPLY_CLAHE=False. La fonction originale n'est pas importée depuis le
# notebook afin d'éviter de déclencher les cellules d'entraînement.
def preprocessing_notebook_reference(image_path, mask_path):
    target_size = 224
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossible de lire le masque : {mask_path}")
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    img = img * mask
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('mask', type=Path)
    parser.add_argument('--save', type=Path, default=None,
                        help='Chemin optionnel pour un rapport JSON local')
    args = parser.parse_args()

    reference = preprocessing_notebook_reference(args.image, args.mask)
    production = preprocess_image(args.image, args.mask)
    identical = np.array_equal(reference, production)
    max_diff = float(np.max(np.abs(reference - production)))
    print('=== PREPROCESSING ===')
    print('Notebook :', reference.shape, reference.dtype)
    print('MLOps    :', production.shape, production.dtype)
    print('Égalité exacte :', identical)
    print('Écart absolu maximal :', max_diff)
    print('SHA256 référence :', hashlib.sha256(reference.tobytes()).hexdigest())
    print('SHA256 MLOps     :', hashlib.sha256(production.tobytes()).hexdigest())
    if not identical:
        raise SystemExit('ÉCHEC : les deux preprocessings diffèrent. Pas de validation du modèle.')

    print('\n=== INFÉRENCE ===')
    model = load_covid_model()
    p_ref = np.asarray(predict_image(model, make_batch(reference)), dtype=float)
    p_mlops = np.asarray(predict_image(model, make_batch(production)), dtype=float)
    for idx, label in enumerate(CLASS_NAMES):
        print(f'{label:17s} référence={p_ref[idx]:.8f}  mlops={p_mlops[idx]:.8f}')
    close = np.allclose(p_ref, p_mlops, rtol=1e-5, atol=1e-6)
    print('Probabilités proches :', close)
    print('Classe prédite :', CLASS_NAMES[int(np.argmax(p_mlops))])
    print('Classe réelle annoncée : COVID')
    print('Prédiction correcte :', int(np.argmax(p_mlops)) == CLASS_NAMES.index('COVID'))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'image': str(args.image), 'mask': str(args.mask),
            'preprocessing_identical': bool(identical),
            'max_pixel_difference': max_diff,
            'predictions_close': bool(close),
            'class_names': CLASS_NAMES,
            'reference_probabilities': p_ref.tolist(),
            'mlops_probabilities': p_mlops.tolist(),
            'predicted_class': CLASS_NAMES[int(np.argmax(p_mlops))],
            'true_class': 'COVID',
        }
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Rapport enregistré :', args.save)
    if not close:
        raise SystemExit('ÉCHEC : les sorties numériques diffèrent.')
    print('\nVALIDATION : les deux chemins de preprocessing et inférence concordent.')

if __name__ == '__main__':
    main()
