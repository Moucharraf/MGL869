import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import anomalies_pb2
import os
import pandas as pd
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration des chemins
SCHEMA_PATH = "data/schema.pbtxt"
DATA_DIR = "data"

# Définition des colonnes et contraintes
NUMERIC_COLS = [
    'Age at enrollment',
    'Admission grade',
    'Previous qualification (grade)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)'
]

BINARY_COLS = [
    'Gender',
    'Displaced',
    'Debtor',
    'Scholarship holder',
]

DOMAIN_CONSTRAINTS = {
    'Age at enrollment': {'min': 17.0, 'max': 65.0, 'required': True},
    'Admission grade': {'min': 0.0, 'max': 200.0, 'required': True},
    'Previous qualification (grade)': {'min': 0.0, 'max': 200.0, 'required': True},
    'Curricular units 1st sem (grade)': {'min': 0.0, 'max': 20.0, 'required': False},
    'Curricular units 2nd sem (grade)': {'min': 0.0, 'max': 20.0, 'required': False},
    'Gender': {'values': ['0', '1'], 'required': True},
    'Displaced': {'values': ['0', '1'], 'required': True},
    'Debtor': {'values': ['0', '1'], 'required': True},
    'Scholarship holder': {'values': ['0', '1'], 'required': True}
}


def prepare_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare un DataFrame brut pour la validation TFDV avec conversion minimale

    Args:
        df: DataFrame brut à préparer

    Returns:
        DataFrame préparé avec les types de base
    """
    df_prep = df.copy()

    # Conversion minimale - seulement les types de base
    for col in df_prep.columns:
        # Convertir les colonnes numériques en float
        if col in NUMERIC_COLS:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').astype(float)
        # Convertir les colonnes binaires en string (pour préserver les valeurs originales)
        elif col in BINARY_COLS:
            # Convertir tout en string, y compris les valeurs numériques
            df_prep[col] = df_prep[col].astype(str)

    return df_prep


def generate_and_save_schema(df_raw: pd.DataFrame, schema_path: str = SCHEMA_PATH) -> schema_pb2.Schema:
    """
    Génère et sauvegarde un schéma TFDV à partir d'un DataFrame brut

    Args:
        df_raw: Données brutes à valider
        schema_path: Chemin du fichier de schéma

    Returns:
        Schéma TFDV enrichi
    """
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)

    logger.info("Génération des statistiques TFDV...")
    df_prepared = prepare_for_validation(df_raw)
    stats = tfdv.generate_statistics_from_dataframe(df_prepared)

    logger.info("Inférence du schéma TFDV...")
    schema = tfdv.infer_schema(statistics=stats)

    # Enrichissement du schéma basé sur les contraintes définies
    for feature in schema.feature:
        col_name = feature.name

        if col_name in DOMAIN_CONSTRAINTS:
            constraints = DOMAIN_CONSTRAINTS[col_name]

            # Définir le type et les contraintes de domaine
            if 'min' in constraints and 'max' in constraints:
                # Colonnes numériques
                feature.type = schema_pb2.FLOAT
                feature.float_domain.min = constraints['min']
                feature.float_domain.max = constraints['max']
                logger.info(
                    f"Contraintes numériques pour {col_name}: min={constraints['min']}, max={constraints['max']}")

            elif 'values' in constraints:
                # Colonnes catégorielles/binaires
                feature.type = schema_pb2.BYTES
                # Effacer les valeurs existantes et définir les valeurs autorisées
                del feature.string_domain.value[:]
                feature.string_domain.value.extend(constraints['values'])
                logger.info(f"Valeurs autorisées pour {col_name}: {constraints['values']}")

            # Définir la présence obligatoire
            if constraints.get('required', False):
                feature.presence.min_fraction = 1.0
                feature.presence.min_count = 1
                logger.info(f"Colonne {col_name} définie comme obligatoire")

    logger.info("Schéma enrichi avec contraintes explicites")
    tfdv.write_schema_text(schema, schema_path)
    logger.info(f"Schéma sauvegardé dans : {schema_path}")

    return schema


def validate_new_data(new_df: pd.DataFrame, schema_path: str = SCHEMA_PATH) -> anomalies_pb2.Anomalies:
    """
    Valide un nouveau DataFrame brut contre un schéma TFDV enregistré

    Args:
        new_df: Nouvelles données brutes à valider
        schema_path: Chemin du schéma TFDV

    Returns:
        Résultat de la validation contenant les anomalies
    """
    logger.info("Validation des nouvelles données...")
    new_df_prepared = prepare_for_validation(new_df)

    # Afficher les données préparées pour debug
    logger.info("Données préparées pour validation:")
    logger.info(f"Colonnes: {list(new_df_prepared.columns)}")
    logger.info(f"Types: {new_df_prepared.dtypes.to_dict()}")

    # Afficher quelques exemples de valeurs
    for col in new_df_prepared.columns:
        unique_vals = new_df_prepared[col].dropna().unique()
        logger.info(f"{col}: {unique_vals[:5]}...")  # Première 5 valeurs uniques

    new_stats = tfdv.generate_statistics_from_dataframe(new_df_prepared)

    # Charger le schéma texte
    loaded_schema = tfdv.load_schema_text(schema_path)

    # Afficher le schéma chargé pour debug
    logger.info("Schéma chargé:")
    for feature in loaded_schema.feature:
        logger.info(f"Feature: {feature.name}, Type: {feature.type}")
        if feature.type == schema_pb2.FLOAT:
            logger.info(f"  Float domain: min={feature.float_domain.min}, max={feature.float_domain.max}")
        elif feature.type == schema_pb2.BYTES:
            logger.info(f"  String domain: {list(feature.string_domain.value)}")

    anomalies = tfdv.validate_statistics(statistics=new_stats, schema=loaded_schema)

    if anomalies.anomaly_info:
        logger.error("Anomalies détectées:")
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            logger.error(f"  {feature_name}: {anomaly_info.description}")
        # Afficher les anomalies avec TFDV si disponible
        try:
            tfdv.display_anomalies(anomalies)
        except:
            logger.warning("Impossible d'afficher les anomalies avec tfdv.display_anomalies")
    else:
        logger.info("Aucune anomalie détectée - données valides")

    return anomalies


def load_data(file_path: str, sep: str = ',') -> pd.DataFrame:
    """Charge les données depuis un fichier CSV avec gestion d'erreurs"""
    try:
        logger.info(f"Chargement des données depuis {file_path}")
        return pd.read_csv(file_path, sep=sep)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Charger les données brutes depuis CSV
        data_path = os.path.join(DATA_DIR, "sample_100_without_target.csv")
        df_train = load_data(data_path)

        # Générer et sauvegarder le schéma
        schema = generate_and_save_schema(df_train)

    except Exception as e:
        logger.exception("Une erreur critique s'est produite lors de la validation")