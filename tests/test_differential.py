import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class TestDifferentialTesting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Obtenir le chemin absolu du fichier de données
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        data_path = os.path.join(root_dir, 'data', 'data.csv')

        # Charger et préprocesser les données
        cls.data_processor = DataProcessor()
        cls.df = cls.data_processor.load_data(data_path)

        # Préparation des données pour l'entraînement avec traitement des NaN
        cls.df['Target'] = cls.df['Target'].map({'Dropout': 1, 'Graduate': 0}).fillna(0)

        # Séparation train/test avant le prétraitement
        X_train, X_test, cls.y_train, cls.y_test = train_test_split(
            cls.df.drop(columns=['Target']), cls.df['Target'],
            test_size=0.2, random_state=42, stratify=cls.df['Target']
        )

        # Prétraitement des données d'entraînement et de test
        train_data = X_train.copy()
        train_data['Target'] = cls.y_train
        cls.X_train_processed = cls.data_processor.fit_transform(train_data).drop(columns=['Target', 'Target_encoded'])

        test_data = X_test.copy()
        test_data['Target'] = cls.y_test
        processed_test = cls.data_processor.transform(test_data)
        cls.X_test_processed = processed_test.drop(columns=['Target', 'Target_encoded'])

        # S'assurer qu'il n'y a pas de NaN dans les données
        cls.X_train_processed = cls.X_train_processed.fillna(0)
        cls.X_test_processed = cls.X_test_processed.fillna(0)
        cls.y_train = cls.y_train.fillna(0)
        cls.y_test = cls.y_test.fillna(0)

    def evaluate_model(self, model, name):
        """Évalue un modèle et retourne ses métriques"""
        model.fit(self.X_train_processed, self.y_train)
        y_pred = model.predict(self.X_test_processed)

        return {
            'name': name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }

    def test_model_comparison(self):
        """Compare les performances de différents modèles"""
        # Définir les modèles à comparer
        models = [
            (GradientBoostingClassifier(random_state=42), "GradientBoosting"),
            (RandomForestClassifier(random_state=42), "RandomForest"),
            (xgb.XGBClassifier(random_state=42), "XGBoost")
        ]

        # Évaluer chaque modèle
        results = []
        for model, name in models:
            metrics = self.evaluate_model(model, name)
            results.append(metrics)
            print(f"\nRésultats pour {name}:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-score: {metrics['f1']:.3f}")

        # Vérifier que GradientBoosting est compétitif
        gb_results = next(r for r in results if r['name'] == "GradientBoosting")
        other_models_f1 = [r['f1'] for r in results if r['name'] != "GradientBoosting"]

        self.assertGreaterEqual(gb_results['f1'], min(other_models_f1),
                              "GradientBoosting n'est pas compétitif avec les autres modèles")

    def test_prediction_consistency(self):
        """Vérifie la cohérence des prédictions entre les modèles"""
        # Entraîner les modèles
        gb = GradientBoostingClassifier(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        xgb_model = xgb.XGBClassifier(random_state=42)

        models = [gb, rf, xgb_model]
        for model in models:
            model.fit(self.X_train_processed, self.y_train)

        # Obtenir les prédictions
        predictions = []
        for model in models:
            y_pred = model.predict(self.X_test_processed)
            predictions.append(y_pred)

        # Calculer le taux d'accord entre les modèles
        agreement_rate = np.mean([
            np.mean(predictions[0] == predictions[1]),
            np.mean(predictions[1] == predictions[2]),
            np.mean(predictions[0] == predictions[2])
        ])

        # Le taux d'accord devrait être supérieur à 80%
        self.assertGreater(agreement_rate, 0.8,
                          "Les modèles ne sont pas suffisamment cohérents entre eux")

    def test_model_stability(self):
        """Test de la stabilité des prédictions entre différents modèles"""
        # Initialisation des modèles
        gb = GradientBoostingClassifier(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        xgb_model = xgb.XGBClassifier(random_state=42)

        models = [gb, rf, xgb_model]
        for model in models:
            model.fit(self.X_train_processed, self.y_train)

        # Obtenir les prédictions
        predictions = []
        for model in models:
            y_pred = model.predict(self.X_test_processed)
            predictions.append(y_pred)

        # Vérifier la stabilité des prédictions
        for i in range(1, len(predictions)):
            # Le taux de concordance entre les modèles devrait être supérieur à 90%
            concordance_rate = np.mean(predictions[0] == predictions[i])
            self.assertGreater(concordance_rate, 0.9,
                               f"Les prédictions du modèle {i} ne sont pas suffisamment stables par rapport au modèle 0")

if __name__ == '__main__':
    unittest.main()
