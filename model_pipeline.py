# modele_pipeline.py
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import logging

from data_processor_transformer import DataProcessorTransformer
from config import *

class StudentDropoutPipeline:
    """
    Pipeline complète pour la prédiction du décrochage scolaire.
    Gère l'entraînement, la prédiction, l'évaluation, la sauvegarde et le chargement du modèle.
    """
    def __init__(self):
        self.pipeline = None
        self.target_encoder = LabelEncoder()
        self.feature_names = None
        self.logger = logging.getLogger("StudentDropoutPipeline")

    def create_pipeline(self):
        self.logger.info("Création de la pipeline scikit-learn...")
        self.pipeline = Pipeline([
            ('preprocessor', DataProcessorTransformer()),
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(**MODEL_PARAMS))
        ])
        return self.pipeline

    def train(self, df):
        """Entraîne la pipeline sur les données fournies."""
        if 'Target' not in df.columns:
            raise ValueError("La colonne 'Target' est manquante.")
        df = df[df["Target"] != "Enrolled"].copy()
        X = df.drop(columns=['Target'])
        y = df['Target']
        y_encoded = self.target_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
        self.create_pipeline()
        self.logger.info(f"Début de l'entraînement sur {X_train.shape[0]} exemples...")
        self.pipeline.fit(X_train, y_train)
        self.feature_names = self.pipeline.named_steps['preprocessor'].feature_names
        self.logger.info("Entraînement terminé. Features utilisées : %s", self.feature_names)
        return X_test, y_test

    def predict(self, X):
        self.logger.info(f"Prédiction sur {len(X)} exemples...")
        preds = self.pipeline.predict(X)
        return self.target_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        self.logger.info("Évaluation du modèle...")
        preds = self.pipeline.predict(X_test)
        # Convertir les classes en chaînes de caractères si nécessaire
        target_names = [str(cls) for cls in self.target_encoder.classes_]
        report = classification_report(
            y_test, preds, target_names=target_names
        )
        return {
            'accuracy': accuracy_score(y_test, preds),
            'recall': recall_score(y_test, preds, average='binary'),
            'f1_score': f1_score(y_test, preds, average='binary'),
            'classification_report': report
        }

    def save_model(self, path):
        self.logger.info(f"Sauvegarde du modèle dans {path} ...")
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'target_encoder': self.target_encoder,
                'feature_names': self.feature_names
            }, f)
        self.logger.info("Modèle sauvegardé.")

    def load_model(self, path):
        self.logger.info(f"Chargement du modèle depuis {path} ...")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.pipeline = obj['pipeline']
            self.target_encoder = obj['target_encoder']
            self.feature_names = obj['feature_names']
        self.logger.info("Modèle chargé.")

    def get_feature_importance(self):
        classifier = self.pipeline.named_steps['classifier']
        importances = classifier.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)


def main():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH, sep=';')
    model_pipeline = StudentDropoutPipeline()

    try:
        X_test, y_test = model_pipeline.train(df)
        print("Évaluation du modèle...")
        results = model_pipeline.evaluate(X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1-score: {results['f1_score']:.3f}")
        print("\nRapport de classification:\n", results['classification_report'])

        print("\nTop 10 des features les plus importantes:")
        print(model_pipeline.get_feature_importance().head(10))

        print(f"\nSauvegarde du modèle dans {MODEL_PATH}")
        model_pipeline.save_model(MODEL_PATH)

    except ValueError as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()