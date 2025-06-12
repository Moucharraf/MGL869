# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY . ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par le serveur
EXPOSE 5000

# Commande pour lancer le serveur
CMD ["python", "server.py"]

