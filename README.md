# Projet de Session - MGL869

## Description
Ce projet vise à prédire l'abandon scolaire en fonction du profil des étudiants. Il s'appuie sur un pipeline de traitement des données, un modèle de machine learning, et une interface utilisateur statique. Le projet inclut des scripts pour le traitement et la transformation des données, l'entraînement du modèle, ainsi que des tests pour le monitoring et la validation des différences. Un serveur permet de déployer le modèle pour une utilisation interactive.

## Structure du Projet
```
config.py
data_processor_transformer.py
data_processor.py
data_validation.py
Dockerfile
model_pipeline.py
README.md
requirements.txt
server.py
data/
    data.csv
    sample_100_without_target.csv
    schema.pbtxt
models/
    gradient_boosting_pipeline.pkl
static/
    index.html
tests/
    test_data_validation.py 
    test_differential.py
    test_monitoring.py
```

- **config.py** : Fichier de configuration pour les paramètres globaux.
- **data_processor_transformer.py** : Script pour transformer les données.
- **data_processor.py** : Script pour le traitement des données.
- **data_validation.py** : Script de validation des données avec TensorFlow Data Validation (TFDV).
- **model_pipeline.py** : Contient le pipeline du modèle de machine learning.
- **server.py** : Serveur pour déployer le modèle.
- **data/** : Contient les données d'entrée.
- **models/** : Contient les modèles entraînés.
- **static/** : Contient les fichiers statiques pour l'interface utilisateur.
- **tests/** : Contient les tests pour le monitoring, les différences et la validation des données.

## Dossier Data

Le dossier `data/` contient toutes les données d'entrée ainsi que le schéma de validation utilisé dans le projet.

- `data.csv` : Dataset complet utilisé pour l'entraînement et la validation.
- `sample_100_without_target.csv` : Échantillon de 100 lignes (sans la variable cible), utilisé pour la validation des données et aussi pour tester la prediction par lot.
- `schema.pbtxt` : Schéma de validation des données au format TensorFlow Data Validation (TFDV).


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
Exécutez tous les tests avec la commande suivante :
```bash
python -m unittest discover tests -v
```
Pour des tests spécifiques (exemple : monitoring), utilisez :
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
3. Accédez à l'application via `http://localhost:5000`.

## Auteur
- Maazou Naboko Mouhamed Moucharraf

## Contributeurs
- Yahyani Mohamed
- Mohamed Tine Seyidina
- Youssoufi Garba Abdourrahmane
