# Classification de radiographies COVID-19 — Projet MLOps

[![Python](https://img.shields.io/badge/Python-3.12.3-blue)](#environnement-technique)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-orange)](#environnement-technique)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](#api-fastapi)
[![Docker](https://img.shields.io/badge/Docker-v1-2496ED)](#exécution-avec-docker)

> **Avertissement médical :** cette application est un prototype pédagogique de classification d'images thoraciques. Elle **ne constitue pas un dispositif médical** et ne doit pas servir à poser un diagnostic, à orienter une prise en charge ou à prendre une décision clinique. Les scores de sortie ne sont pas des probabilités cliniques calibrées.

## 1. Présentation et objectifs

Ce dépôt est la partie **MLOps** d'un projet de Data Science consacré à la classification de radiographies pulmonaires à partir de la *COVID-19 Radiography Database* (21 165 images réparties en quatre catégories). Il reprend un modèle **VGG16 fine-tuné**, déjà entraîné, et le transforme en service d'inférence **FastAPI** exécutable localement ou dans **Docker**.

La **phase 1 — Fondations et conteneurisation** vise à :

1. sélectionner et identifier un modèle entraîné ;
2. isoler un environnement Python reproductible ;
3. extraire et tester le prétraitement initialement défini dans un notebook ;
4. valider l'inférence avec des tests unitaires, d'intégration et de non-régression ;
5. exposer un endpoint HTTP de prédiction ;
6. lancer l'ensemble dans un conteneur Docker.

**Périmètre :** inférence locale, une seule API, traitement d'une radiographie et de son **masque pulmonaire correspondant**. La segmentation automatique, le suivi des expériences, l'orchestration et le monitoring avancé ne font pas partie de cette version.

## 2. Modèle et résultats de référence

| Propriété | Valeur |
|---|---|
| Architecture | VGG16 (transfer learning, tête de classification personnalisée) |
| Artefact | `models/final_vgg16_fixed_splits.keras` |
| Format | Keras natif `.keras` |
| Entrée réseau | `224 × 224 × 3`, `float32`, batch `(N,224,224,3)` |
| Sortie | 4 scores Softmax |
| CLAHE | Désactivé |
| Data augmentation | Désactivée pour la variante sélectionnée |
| Masque pulmonaire | **Obligatoire** |
| Nombre de paramètres | 14 847 044 |
| Accuracy de référence (jeu de test) | **0,886** |
| Macro-F1 de référence (jeu de test) | **0,880** |
| Recall COVID de référence (jeu de test) | **0,810** |
| Identifiant applicatif du modèle | `v1.0` |

Ces métriques proviennent des évaluations Data Science réalisées avant le déploiement ; **elles ne mesurent pas la qualité du service HTTP** et ne constituent pas une validation clinique externe.

**Ordre impératif des classes :**

| Indice | Classe |
|---:|---|
| 0 | `COVID` |
| 1 | `Normal` |
| 2 | `Lung_Opacity` |
| 3 | `Viral Pneumonia` |

**Empreinte SHA256 de l'artefact validé :**

```text
ec80d4bc143d2e8d4a8bf2d08e7ca3db789b8164cdf8adc34dcba4d38050544a
```

Après avoir placé le modèle dans `models/`, vérifier son intégrité :

```bash
sha256sum models/final_vgg16_fixed_splits.keras
```

Le fichier `.keras` est **exclu du dépôt Git**. Il doit être fourni séparément, depuis un emplacement de confiance, avec l'empreinte ci-dessus. Il n'est pas nécessaire de déployer les plusieurs gigaoctets du dataset pour faire fonctionner le service d'inférence.

## 3. Architecture

```text
Client HTTP / Swagger
       |
       | POST /predict (image + mask)
       v
   FastAPI (app/main.py)
       |
       +--> validation des fichiers (format, taille, décodage)
       |
       v
   app/preprocessing.py
       |   grayscale -> masque -> resize -> RGB -> float32
       v
   make_batch()  [1, 224, 224, 3]
       |
       v
   app/model.py
       |   VGG16 .keras (chargé au démarrage)
       |   preprocess_input VGG16 déjà intégré dans le modèle
       v
   Softmax : 4 scores
       |
       v
   Réponse JSON
```

**Ne pas appliquer deux fois `vgg16.preprocess_input` ni normaliser les images en `[0,1]`** : cette transformation est déjà incluse dans l'artefact enregistré. Les tableaux fournis au modèle sont en `float32`, sur l'échelle des pixels `[0,255]`.

### Structure du dépôt

```text
covid19-radiography-mlops/
├── app/
│   ├── __init__.py
│   ├── main.py                 # API FastAPI
│   ├── model.py                # Chargement et inférence VGG16
│   ├── preprocessing.py        # Préparation image + masque
│   ├── config.py               # Réservé à la configuration ultérieure
│   └── schemas.py              # Réservé aux schémas ultérieurs
├── models/
│   └── final_vgg16_fixed_splits.keras  # LOCAL, non versionné dans Git
├── data/
│   └── samples/
│       ├── covid_test.png      # Échantillon local de validation
│       └── covid_mask.png      # Masque correspondant
├── tests/
│   ├── test_preprocessing.py
│   ├── test_integration.py
│   └── test_api.py
├── scripts/
│   ├── validate_model.py
│   ├── validate_real_image.py
│   ├── compare_notebook_mlops.py
│   └── benchmark.py            # Réservé au benchmark
├── requirements.txt
├── requirements-dev.txt        # Option de séparation future des dépendances
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

Les scripts facultatifs et les fichiers réservés à de futures évolutions peuvent être vides ou absents selon l'état du dépôt : ils ne sont pas requis par le serveur Docker.

## 4. Environnement technique

Configuration validée sur **Ubuntu 24.04** :

| Composant | Version testée |
|---|---|
| Python | 3.12.3 |
| TensorFlow | 2.17.1 |
| Keras | 3.14.1 |
| NumPy | 1.26.4 |
| OpenCV headless | 4.10.0.84 |
| Pytest | 9.1.1 |

Les versions de FastAPI, Uvicorn, Starlette et des dépendances transitives sont celles enregistrées dans le `requirements.txt` du dépôt après validation. **Ne pas installer sans contrainte la dernière version d'OpenCV** : une tentative d'installation d'OpenCV 5 avait entraîné NumPy 2.x, incompatible avec TensorFlow 2.17.1.

### Installation locale

Dans un terminal ouvert à la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Le test de compatibilité attendu est :

```text
No broken requirements found.
```

Installer et utiliser uniquement les dépendances prévues pour le projet. Si `requirements.txt` a été généré avec `pip freeze`, il constitue l'instantané de l'environnement testé, mais peut inclure des outils de développement : une séparation production/développement est prévue ultérieurement.

## 5. Prétraitement des radiographies

L'API attend **une radiographie** et **son masque pulmonaire associé**. Les deux fichiers doivent représenter le même examen.

Le module `app/preprocessing.py` :

1. lit l'image et le masque en niveaux de gris ;
2. redimensionne le masque aux dimensions de l'image avec `cv2.INTER_NEAREST` ;
3. binarise le masque par `(mask > 0)` ;
4. applique le masque à l'image ;
5. redimensionne le résultat à `224 × 224` avec `cv2.INTER_AREA` ;
6. convertit l'image de gris vers RGB ;
7. convertit en `float32` et ajoute la dimension batch.

Le modèle possède déjà une couche interne `vgg16_preprocess`. **Le preprocessing n'applique ni CLAHE, ni augmentation aléatoire, ni normalisation `[0,1]`.**

Les masques étaient fournis par le dataset d'origine. Pour des radiographies externes sans masque, cette première version de l'API **ne peut pas effectuer une inférence conforme au protocole d'entraînement**. Un module de segmentation automatique et sa validation seront nécessaires avant de supprimer cette exigence.

## 6. Validation locale du modèle

Placer l'artefact `.keras` dans le dossier `models/` et un couple image/masque correspondant dans `data/samples/`.

Tester le chargement et l'inférence sur image artificielle :

```bash
python -m scripts.validate_model
```

Tester l'inférence sur une radiographie réelle :

```bash
python -m scripts.validate_real_image \
  data/samples/covid_test.png \
  data/samples/covid_mask.png
```

Comparer le prétraitement de référence issu du notebook avec le module MLOps :

```bash
python -m scripts.compare_notebook_mlops \
  data/samples/covid_test.png \
  data/samples/covid_mask.png \
  --save data/comparison_report.json
```

**Résultat observé :** égalité exacte des deux tableaux prétraités (`SHA256` identique, écart maximal `0.0`) et scores d'inférence concordants sur l'échantillon testé. Ce contrôle porte sur la transcription du prétraitement de référence exécutée dans le projet MLOps, et non sur une validation exhaustive de toutes les images.

### Tests automatisés

```bash
python -m pytest -v
```

Neuf tests ont réussi lors des validations du projet :

| Groupe | Tests | Ce qui est vérifié |
|---|---:|---|
| `test_preprocessing.py` | 3 | Dimensions, masque, déterminisme, entrées absentes |
| `test_integration.py` | 3 | Contrat de sortie, comparaison aux valeurs de référence, stabilité du comportement constaté |
| `test_api.py` | 3 | Endpoints, format de réponse, rejet des uploads invalides (modèle simulé) |

Les tests d'intégration nécessitent **le modèle local et les deux images de test**. La réussite des tests HTTP avec modèle simulé ne remplace pas un essai réel via `/predict`.

## 7. API FastAPI

Démarrer localement (depuis la racine du projet) :

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Documentation interactive : **http://127.0.0.1:8000/docs**

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Présentation du service |
| `GET` | `/health` | Vérifie que le modèle a été chargé (`503` sinon) |
| `GET` | `/model/info` | Nom, version applicative, dimensions, classes, exigence du masque |
| `POST` | `/predict` | Classification d'une image avec son masque |

### Requête d'inférence

Le endpoint `POST /predict` attend un formulaire **`multipart/form-data`** avec deux champs obligatoires :

- `image` : radiographie PNG ou JPEG ;
- `mask` : masque pulmonaire correspondant, PNG ou JPEG.

Taille maximale : **10 Mio par fichier**. Les images doivent être décodables ; l'API rejette également les images au-delà de **20 millions de pixels**.

Exemple avec `curl` :

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -F 'image=@data/samples/covid_test.png;type=image/png' \
  -F 'mask=@data/samples/covid_mask.png;type=image/png'
```

Exemple de résultat **réellement observé** pour notre radiographie de test (classe réelle COVID, mais prédite Lung_Opacity) :

```json
{
  "predicted_class": "Lung_Opacity",
  "confidence": 0.9447658658027649,
  "probabilities": {
    "COVID": 0.05521446466445923,
    "Normal": 0.000017363076040055603,
    "Lung_Opacity": 0.9447658658027649,
    "Viral Pneumonia": 0.000002312724745934247
  }
}
```

L'API renvoie **également** `model_version`, `inference_ms` et un avertissement de non-usage clinique dans son schéma complet. Les scores ci-dessus sont un extrait des champs communiqués pendant la validation HTTP.

**Codes d'erreur usuels :** `413` (fichier > 10 Mio), `415` (type MIME non autorisé), `422` (fichier vide, illisible ou trop grand en pixels), `503` (modèle indisponible), `500` (sortie modèle invalide).

L'API emploie des fichiers temporaires pour réutiliser le preprocessing existant ; ils sont supprimés automatiquement après traitement. La version actuelle n'offre ni authentification, ni limitation de débit, ni garanties de sécurité appropriées à des données médicales sensibles.

## 8. Exécution avec Docker

### Construction

Docker doit être installé et **un moteur Docker accessible**. Exécuter les commandes ci-dessous depuis la racine du projet :

```bash
docker build -t covid19-api:v1 .
```

Le `Dockerfile` utilise `python:3.12-slim`, installe `requirements.txt` et copie uniquement le code de `app/`. Le **modèle n'est pas copié dans l'image** : il est monté en lecture seule à l'exécution.

### Démarrage

```bash
docker run --rm \
  --name covid19-api \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  covid19-api:v1
```

Ouvrir **http://127.0.0.1:8000/docs**, tester `GET /health`, puis `POST /predict` avec le même couple image/masque. Le port est lié à l'interface locale uniquement.

Le serveur Python local et le conteneur Docker ne peuvent pas utiliser simultanément le même port `127.0.0.1:8000` : arrêter l'un avant de lancer l'autre.

### Contrôles Docker

```bash
docker ps
docker logs covid19-api
docker image ls covid19-api
docker stats --no-stream covid19-api
```

Aucune commande de purge Docker n'est nécessaire pour ce projet.

## 9. Mesures réellement observées

| Indicateur | Valeur observée | Remarque |
|---|---|---|
| Image | `covid19-api:v1` | ID `43c573074a3d` |
| Docker `DISK USAGE` | `3.19GB` | Mesure Docker affichée |
| Docker `CONTENT SIZE` | `790MB` | Autre indicateur Docker, non additionnable au précédent |
| CPU | `0.31 %` | Capture instantanée, pas un test de charge |
| RAM conteneur | `402.8 MiB` | Capture instantanée |
| PIDs | `39` | Capture instantanée |
| Tests unitaires + intégration | `6 passed` | Exécution groupée observée |
| Tests FastAPI | `3 passed` | Exécution observée séparément |
| Test HTTP réel | Réponse cohérente avec la référence | FastAPI + VGG16 |

**Non encore mesurés de façon robuste :** latence p50/p95 de `/predict`, requêtes par seconde, consommation mémoire en pic, disponibilité dans la durée. Les valeurs CPU/RAM ci-dessus ne sont **pas** des garanties de performance.

### Cas de mauvaise classification connu

Notre image `covid_test.png` appartient réellement à la classe **COVID**, mais le modèle prédit **Lung_Opacity** (environ **94,48 %** de score Softmax). Le prétraitement de référence et le module MLOps donnent exactement le même tableau ; leurs prédictions concordent. Le test de non-régression **fige ce comportement observé sans le présenter comme une prédiction médicalement correcte**.

## 10. Reproductibilité, versioning et précautions

- Le code est versionné par **Git** ; une étiquette `v1.0.0` peut identifier cette phase une fois le commit réalisé.
- Le modèle est fourni **hors Git** et identifié par son **SHA256** ; un simple numéro de version ne suffit pas à garantir l'identité des poids.
- Le `requirements.txt` doit correspondre aux dépendances **effectivement validées** ; tout changement de versions demande une nouvelle exécution des tests.
- Conserver l'ordre des quatre classes et le preprocessing d'origine.
- Protéger les images personnelles et les données de santé : privilégier des échantillons anonymisés/autorisés, ne pas publier de radiographies identifiantes.
- Ne pas exposer cette API sur Internet ou à des utilisateurs non autorisés dans son état actuel.

## 11. Roadmap MLOps

| Phase | Périmètre | État |
|---|---|---|
| **1 — Fondations & conteneurisation** | Modèle, environnement, preprocessing, tests, FastAPI, Docker | **Prototype fonctionnel validé localement** |
| 2 — Suivi & versioning | MLflow, versioning données/modèles, réflexion microservices | À venir |
| 3 — Orchestration & déploiement | Pipeline CI, sécurisation, montée en charge Docker/Kubernetes | À venir |
| 4 — Monitoring & maintenance | Prometheus/Grafana, Evidently, mises à jour, documentation d'exploitation | À venir |

**Prochains travaux avant toute utilisation plus large :** benchmark des latences, tests de charge, mécanisme de segmentation pulmonaire pour les images sans masque, validation sur des données externes indépendantes, sécurisation et surveillance du service.

---

**État du projet :** phase 1 — premier déploiement local FastAPI + Docker réalisé et testé. **Usage pédagogique et expérimental uniquement.**
