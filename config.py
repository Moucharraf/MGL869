# -*- coding: utf-8 -*-
"""
Configuration file for the student dropout prediction model
"""

import os

# Paths
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/gradient_boosting_pipeline.pkl"
SCHEMA_PATH = "models/schema.tfrecord"
PROCESSED_DATA_PATH = "data/dataset_final_train.csv"

# Data processing parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 150,  # Augmenté pour plus de stabilité
    'learning_rate': 0.03,  # Réduit davantage pour des prédictions plus nuancées
    'max_depth': 2,  # Réduit pour éviter le surapprentissage
    'min_samples_leaf': 20,  # Augmenté pour plus de stabilité
    'min_samples_split': 50,  # Ajouté pour plus de généralisation
    'subsample': 0.7,  # Réduit pour plus de régularisation
    'random_state': RANDOM_STATE
}

# Feature engineering parameters
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.8

# Categorical variables
CATEGORICAL_VARS = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

# Numerical variables
NUMERICAL_VARS = [
    'Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# Variables for outlier detection
CONTINUOUS_VARS = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# Ordinal and binary variables for encoding
ORDINAL_VARS = ['Previous qualification', "Mother's qualification", "Father's qualification"]
BINARY_VARS = ['Gender', 'Displaced', 'Educational special needs', 'Debtor',
               'Tuition fees up to date', 'Scholarship holder', 'International',
               'Daytime/evening attendance']
HIGH_CARDINALITY_VARS = ['Course', 'Application mode', 'Nacionality',
                        "Mother's occupation", "Father's occupation"]

# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs("data", exist_ok=True)