import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from scipy import stats
from data_processor import DataProcessor
from model_pipeline import StudentDropoutPipeline

"""
Test Suite pour le Monitoring du Modèle de Prédiction du Décrochage

Périodes d'analyse :
    - Période 1 (Entraînement) : Premiers 2032 étudiants (45.9% des données)
    - Période 2 (Test) : 2392 étudiants restants (54.1% des données)
    Total du dataset : 4424 étudiants

Seuils de monitoring :
    1. Dérive des données :
       - Test de Kolmogorov-Smirnov avec α = 0.05
       - Tolérance : max 20% des features peuvent présenter une dérive significative
    
    2. Stabilité des performances :
       - Δ Accuracy max : 10% entre les périodes
       - Δ Recall max : 15% entre les périodes
    
    3. Distribution des prédictions :
       - Max 50% de probabilités extrêmes (<0.1 ou >0.9)
       - Δ Taux de prédictions positives max : 15% entre périodes
"""

class TestMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Charger les données (4424 exemples au total)
        cls.data_processor = DataProcessor()
        data = cls.data_processor.load_data('data/data.csv')

        # Encoder la target et sauvegarder une copie
        data['Target'] = data['Target'].map({'Dropout': 1, 'Graduate': 0}).fillna(0)
        cls.df = data.copy()

        # Diviser en deux périodes
        cls.period1_data = data.iloc[:2032].copy()  # 45.9% - Période d'entraînement
        cls.period2_data = data.iloc[2032:].copy()  # 54.1% - Période de test

        # Entraîner le modèle sur la première période
        cls.pipeline = StudentDropoutPipeline()

        # Séparer features et target pour la première période
        X_train = cls.period1_data.drop('Target', axis=1)
        y_train = cls.period1_data['Target']

        # Entraîner le modèle
        cls.pipeline.train(cls.period1_data)

    def test_data_drift(self):
        """Test pour détecter la dérive des données"""
        # Identifier les colonnes numériques
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        drift_detected = []

        for col in numeric_cols:
            if col not in ['Target', 'Target_encoded']:
                # Test de Kolmogorov-Smirnov pour comparer les distributions
                period1_values = self.period1_data[col].fillna(0).values
                period2_values = self.period2_data[col].fillna(0).values

                statistic, p_value = stats.ks_2samp(period1_values, period2_values)

                if p_value < 0.05:  # Seuil de significativité standard
                    drift_detected.append(col)

        # Le nombre de features avec dérive ne devrait pas dépasser 20% du total
        max_acceptable_drift = len(numeric_cols) * 0.2
        self.assertLessEqual(len(drift_detected), max_acceptable_drift,
                           f"Trop de caractéristiques présentent une dérive significative: {drift_detected}")

    def test_performance_stability(self):
        """Test de la stabilité des performances du modèle dans le temps"""
        # Données d'entraînement (première période)
        X_train = self.period1_data.drop('Target', axis=1)
        y_train = self.period1_data['Target'].astype(int)

        # Données de test (seconde période)
        X_test = self.period2_data.drop('Target', axis=1)
        y_test = self.period2_data['Target'].astype(int)

        # Évaluer sur les deux périodes
        results_train = self.pipeline.evaluate(X_train, y_train)
        results_test = self.pipeline.evaluate(X_test, y_test)

        # La différence d'accuracy ne devrait pas dépasser 10%
        accuracy_diff = abs(results_train['accuracy'] - results_test['accuracy'])
        self.assertLess(accuracy_diff, 0.1,
                       "Trop grande différence de performance entre les périodes")

        # La différence de recall ne devrait pas dépasser 15%
        recall_diff = abs(results_train['recall'] - results_test['recall'])
        self.assertLess(recall_diff, 0.15,
                       "Trop grande différence de recall entre les périodes")

    def test_prediction_distribution(self):
        """Test de la distribution des prédictions"""
        # Obtenir les prédictions pour les deux périodes
        period1_pred = self.pipeline.predict(self.period1_data.drop('Target', axis=1))
        period2_pred = self.pipeline.predict(self.period2_data.drop('Target', axis=1))

        # Convertir en numérique et s'assurer qu'il n'y a pas de NaN
        period1_pred = pd.Series(period1_pred).map({'Dropout': 1, 'Graduate': 0}).fillna(0)
        period2_pred = pd.Series(period2_pred).map({'Dropout': 1, 'Graduate': 0}).fillna(0)

        # Calculer les taux de prédictions positives
        rate_period1 = np.mean(period1_pred)
        rate_period2 = np.mean(period2_pred)

        # La différence des taux ne devrait pas dépasser 15%
        rate_difference = abs(rate_period1 - rate_period2)
        self.assertLess(rate_difference, 0.15,
                       f"Changement significatif dans la distribution des prédictions : {rate_difference}")

        # Obtenir et traiter les probabilités
        proba_period2 = self.pipeline.predict_proba(self.period2_data.drop('Target', axis=1))
        if proba_period2 is not None:  # Vérifier si le modèle supporte predict_proba
            proba_period2 = proba_period2[:, 1]  # Probabilité de la classe positive
            extreme_probas = np.mean((proba_period2 < 0.1) | (proba_period2 > 0.9))

            # Vérifier que les probabilités ne sont pas trop extrêmes
            self.assertLess(extreme_probas, 0.5,
                           "Trop de prédictions avec des probabilités extrêmes")
