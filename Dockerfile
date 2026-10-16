# Image de base Python
FROM python:3.12-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Paramètres Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copier les modules de l'application
COPY app/ ./app/

# Créer le dossier où sera monté le modèle
RUN mkdir -p /app/models

# Port utilisé par FastAPI
EXPOSE 8000

# Démarrage du serveur
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]