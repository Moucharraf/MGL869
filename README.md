# Projet de Session - MGL869

## Description
Ce projet est une application d'apprentissage automatique qui inclut un pipeline de traitement des données, un modèle de machine learning, et une interface utilisateur statique. Le projet est structuré pour inclure des tests pour le monitoring et les différences, des scripts de traitement des données, et un serveur pour déployer le modèle.

## Structure du Projet
```
config.py
data_processor_transformer.py
data_processor.py
Dockerfile
model_pipeline.py
README.md
requirements.txt
server.py
data/
    data.csv
models/
    gradient_boosting_pipeline.pkl
static/
    index.html
tests/
    test_differential.py
    test_monitoring.py
```

- **Client_test.py** : Script pour tester le client.
- **config.py** : Fichier de configuration pour les paramètres globaux.
- **data_processor_transformer.py** : Script pour transformer les données.
- **data_processor.py** : Script pour le traitement des données.
- **model_pipeline.py** : Contient le pipeline du modèle de machine learning.
- **server.py** : Serveur pour déployer le modèle.
- **data/** : Contient les données d'entrée.
- **models/** : Contient les modèles entraînés.
- **static/** : Contient les fichiers statiques pour l'interface utilisateur.
- **tests/** : Contient les tests pour le monitoring et les différences.

## Prérequis
- Python 3.9 ou supérieur
- Bibliothèques Python listées dans `requirements.txt`
- Docker (pour le déploiement)

## Installation
1. Clonez ce dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
1. Lancez le serveur :
   ```bash
   python server.py
   ```
2. Accédez à l'interface utilisateur via `index.html` dans le dossier `static/`.

## Tests
Exécutez les tests pour le monitoring et les différences avec :
```bash
python -m unittest tests/test_monitoring.py -v
```

## Déploiement avec Docker
1. Construisez l'image Docker :
   ```bash
   docker build -t projet-session .
   ```
2. Lancez un conteneur :
   ```bash
   docker run -p 5000:5000 projet-session
   ```

## Auteurs
- Moucharraf
- Seyidina
- Mohamed
- Abdourrahmame

## Licence
Ce projet est sous licence MIT.
