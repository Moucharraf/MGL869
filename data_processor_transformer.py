from sklearn.base import BaseEstimator, TransformerMixin
from data_processor import DataProcessor
import pandas as pd
import logging


class DataProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper scikit-learn pour intégrer le DataProcessor dans une pipeline.
    Garantit la cohérence des features entre entraînement et prédiction.
    """

    def __init__(self):
        self.processor = DataProcessor()
        self.feature_names = None
        self.logger = logging.getLogger("DataProcessorTransformer")

    def fit(self, X, y=None):
        """
        Entraîne le DataProcessor sur les données d'entraînement.
        X : DataFrame ou array-like
        y : labels (optionnel)
        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if y is not None and 'Target' not in df.columns:
            df['Target'] = y

        df_processed = self.processor.fit_transform(df)
        self.feature_names = [col for col in df_processed.columns if col != 'Target_encoded']
        self.logger.info(f"fit terminé. Features retenues : {self.feature_names}")
        return self

    def transform(self, X):
        """
        Transforme les données en utilisant le DataProcessor entraîné.
        Garantit que les colonnes sont alignées avec celles vues au fit.
        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        df_processed = self.processor.transform(df)

        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.feature_names]
        self.logger.info(f"transform terminé. Shape : {df_processed.shape}")
        return df_processed.values