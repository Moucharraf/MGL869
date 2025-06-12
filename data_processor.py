# data_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import *
import warnings
import logging

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Classe de prétraitement des données pour la prédiction du décrochage scolaire.
    Gère le nettoyage, la création de features, l'encodage, la détection d'outliers, etc.
    """
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()
        self.variance_selector = None
        self.removed_features = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DataProcessor")

    def clean_special_values(self, df):
        """Remplace les valeurs spéciales/codées par des labels explicites."""
        df_clean = df.copy()

        occupation_mapping = {99: 'Unknown/Blank', 90: 'Other_Situation'}
        qualification_mapping = {34: 'Unknown'}

        for var in ["Mother's occupation", "Father's occupation"]:
            if var in df_clean.columns:
                df_clean[var] = df_clean[var].replace(occupation_mapping)

        for var in ["Mother's qualification", "Father's qualification"]:
            if var in df_clean.columns:
                df_clean[var] = df_clean[var].replace(qualification_mapping)

        self.logger.info("Valeurs spéciales nettoyées.")
        return df_clean

    def create_derived_features(self, df):
        """Crée des variables dérivées utiles à la modélisation."""
        df_enhanced = df.copy()

        if "Mother's qualification" in df.columns and "Father's qualification" in df.columns:
            mother = pd.to_numeric(df["Mother's qualification"].replace('Unknown', np.nan), errors='coerce')
            father = pd.to_numeric(df["Father's qualification"].replace('Unknown', np.nan), errors='coerce')
            df_enhanced['Parents_max_education'] = np.nanmax(np.stack([mother, father], axis=1), axis=1)
            df_enhanced['Parents_max_education'] = df_enhanced['Parents_max_education'].fillna(0)

        if 'Curricular units 1st sem (enrolled)' in df.columns:
            e1 = df['Curricular units 1st sem (enrolled)']
            a1 = df['Curricular units 1st sem (approved)']
            ev1 = df['Curricular units 1st sem (evaluations)']
            df_enhanced['Success_rate_1st_sem'] = np.where(e1 > 0, a1 / e1, 0)
            df_enhanced['Evaluation_rate_1st_sem'] = np.where(e1 > 0, ev1 / e1, 0)

        if 'Curricular units 2nd sem (enrolled)' in df.columns:
            e2 = df['Curricular units 2nd sem (enrolled)']
            a2 = df['Curricular units 2nd sem (approved)']
            ev2 = df['Curricular units 2nd sem (evaluations)']
            df_enhanced['Success_rate_2nd_sem'] = np.where(e2 > 0, a2 / e2, 0)
            df_enhanced['Evaluation_rate_2nd_sem'] = np.where(e2 > 0, ev2 / e2, 0)

        if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
            g1 = df['Curricular units 1st sem (grade)']
            g2 = df['Curricular units 2nd sem (grade)']
            df_enhanced['Average_grade'] = np.where((g1 > 0) & (g2 > 0), (g1 + g2) / 2, np.where(g1 > 0, g1, np.where(g2 > 0, g2, 0)))
            df_enhanced['Grade_improvement'] = np.where((g1 > 0) & (g2 > 0), g2 - g1, 0)

        # Facteurs de risque
        risk_factors = []
        if 'Previous qualification (grade)' in df.columns:
            q1 = df['Previous qualification (grade)'].quantile(0.25)
            df_enhanced['Low_previous_grade'] = (df['Previous qualification (grade)'] < q1).astype(int)
            risk_factors.append('Low_previous_grade')

        if 'Debtor' in df.columns:
            df_enhanced['Debtor'] = pd.to_numeric(df['Debtor'], errors='coerce').fillna(0)
            risk_factors.append('Debtor')

        if 'Tuition fees up to date' in df.columns:
            df_enhanced['Tuition_not_uptodate'] = (df['Tuition fees up to date'] == 0).astype(int)
            risk_factors.append('Tuition_not_uptodate')

        df_enhanced['Risk_score'] = df_enhanced[risk_factors].sum(axis=1) if risk_factors else 0
        self.logger.info("Features dérivées créées : %s", list(df_enhanced.columns))
        return df_enhanced

    def encode_categorical_variables(self, df, fit_encoders=True):
        """Encode les variables catégorielles (label, one-hot, etc.)."""
        df_model = df.copy()
        categorical_columns = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'Target' in categorical_columns:
            categorical_columns.remove('Target')

        if 'Target' in df_model.columns:
            if fit_encoders:
                le_target = LabelEncoder()
                df_model['Target_encoded'] = le_target.fit_transform(df_model['Target'])
                self.target_encoder = le_target

        for var in BINARY_VARS:
            if var in df_model.columns:
                df_model[var] = pd.to_numeric(df_model[var], errors='coerce')

        for var in ORDINAL_VARS:
            if var in df_model.columns:
                le = LabelEncoder()
                df_model[f"{var}_encoded"] = le.fit_transform(df_model[var].astype(str))
                self.label_encoders[var] = le

        for var in HIGH_CARDINALITY_VARS:
            if var in df_model.columns and var in categorical_columns:
                top_cats = df_model[var].value_counts().nlargest(10).index
                df_model[f"{var}_top"] = df_model[var].apply(lambda x: x if x in top_cats else 'Other')
                dummies = pd.get_dummies(df_model[f"{var}_top"], prefix=var)
                df_model = pd.concat([df_model, dummies], axis=1)

        drop_cols = categorical_columns + [f"{var}_top" for var in HIGH_CARDINALITY_VARS if f"{var}_top" in df_model.columns]
        if 'Target' in drop_cols:
            drop_cols.remove('Target')
        df_model.drop(columns=[col for col in drop_cols if col in df_model.columns], inplace=True)

        self.logger.info("Variables catégorielles encodées. Colonnes : %s", list(df_model.columns))
        return df_model

    def clean_outliers_winsorization(self, df, vars_list):
        """Applique la winsorisation pour limiter l'impact des outliers."""
        df_winsor = df.copy()
        for col in vars_list:
            if col in df_winsor.columns:
                q1 = df_winsor[col].quantile(0.25)
                q3 = df_winsor[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df_winsor[col] = df_winsor[col].clip(lower, upper)
        self.logger.info("Outliers traités (winsorisation) sur : %s", vars_list)
        return df_winsor

    def remove_correlated_features(self, df, threshold=CORRELATION_THRESHOLD, target_col='Target_encoded'):
        """Supprime les features trop corrélées entre elles."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        corr_matrix = df[numeric_cols].corr().abs()
        to_remove = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    to_remove.add(col1 if df[col1].var() < df[col2].var() else col2)

        self.removed_features = list(to_remove)
        self.logger.info("Features corrélées supprimées : %s", self.removed_features)
        return df.drop(columns=self.removed_features)

    def remove_low_variance(self, df, fit_selector=True):
        """Supprime les features à faible variance."""
        from sklearn.feature_selection import VarianceThreshold

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit_selector:
            self.variance_selector = VarianceThreshold(threshold=0.01)
            self.variance_selector.fit(df[numeric_cols])
            self.variance_numeric_cols = numeric_cols
        else:
            for col in self.variance_numeric_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.variance_numeric_cols]

        selected = df.columns[self.variance_selector.get_support(indices=True)].tolist()
        self.logger.info("Features à faible variance supprimées. Colonnes finales : %s", list(df.columns))
        return df[selected]

    def fit_transform(self, df):
        """Pipeline complète de prétraitement pour l'entraînement."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Entrée non conforme : DataFrame attendu.")
        df = df[df["Target"] != "Enrolled"].copy() if "Target" in df.columns else df.copy()
        df = self.clean_special_values(df)
        df = self.create_derived_features(df)
        df = self.encode_categorical_variables(df, fit_encoders=True)

        cont_vars = [var for var in CONTINUOUS_VARS if var in df.columns]
        cont_vars += [var for var in ['Success_rate_1st_sem', 'Success_rate_2nd_sem', 'Average_grade'] if var in df.columns]

        df = self.clean_outliers_winsorization(df, cont_vars)
        df = self.remove_correlated_features(df)
        df = self.remove_low_variance(df, fit_selector=True)
        self.logger.info("fit_transform terminé. Shape finale : %s", df.shape)
        return df

    def transform(self, df):
        """Pipeline de prétraitement pour la prédiction (inférence)."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Entrée non conforme : DataFrame attendu.")
        df = self.clean_special_values(df)
        df = self.create_derived_features(df)
        df = self.encode_categorical_variables(df, fit_encoders=False)

        cont_vars = [var for var in CONTINUOUS_VARS if var in df.columns]
        cont_vars += [var for var in ['Success_rate_1st_sem', 'Success_rate_2nd_sem', 'Average_grade'] if var in df.columns]

        df = self.clean_outliers_winsorization(df, cont_vars)
        df.drop(columns=[col for col in self.removed_features if col in df.columns], inplace=True)
        df = self.remove_low_variance(df, fit_selector=False)
        self.logger.info("transform terminé. Shape finale : %s", df.shape)
        return df

    def load_data(self, data_path):
        """Charge les données à partir d'un fichier CSV avec un séparateur spécifique."""
        try:
            df = pd.read_csv(data_path, sep=';')
            self.logger.info(f"Données chargées avec succès depuis {data_path}")
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données : {e}")
            raise
