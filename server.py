from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Charger le modèle
model_obj = joblib.load('models/gradient_boosting_pipeline.pkl')
if isinstance(model_obj, dict):
    model = model_obj.get('model') or model_obj.get('pipeline')
else:
    model = model_obj

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Conversion automatique des champs numériques
        numeric_fields = [
            'Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
            'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
        for field in numeric_fields:
            if field in data and data[field] != '':
                if '.' in str(data[field]) or 'e' in str(data[field]).lower():
                    data[field] = float(data[field])
                else:
                    data[field] = int(data[field])

        input_data = pd.DataFrame([data])

        # Prédiction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Explicabilité avec SHAP
        try:
            # Extraire le classifieur du pipeline
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                # Transformer les données avec le preprocesseur
                X_trans = preprocessor.transform(input_data)
            else:
                classifier = model
                X_trans = input_data.values

            # Créer l'explainer et calculer les valeurs SHAP
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans)

            # Pour les modèles de classification binaire
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Prendre les valeurs pour la classe positive

            # Création du graphique SHAP
            plt.figure(figsize=(12, 6))
            plt.clf()

            # Obtenir les 5 variables les plus influentes
            feature_importance = list(zip(input_data.columns, shap_values[0]))
            top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:5]
            feature_names, feature_values = zip(*top_features)

            # Créer le graphique à barres horizontales
            y_pos = range(len(feature_names))
            bars = plt.barh(y_pos, feature_values)

            # Personnalisation du graphique
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Impact sur la prédiction')
            plt.title('Variables les plus influentes dans la prédiction')

            # Colorer les barres selon l'impact
            for i, bar in enumerate(bars):
                if feature_values[i] > 0:
                    bar.set_color('#ff6b6b')  # Rouge pour impact positif
                else:
                    bar.set_color('#4dabf7')  # Bleu pour impact négatif

            # Ajouter une grille
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)

            # Sauvegarder le graphique
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='none')
            plt.close()
            buf.seek(0)
            explain_plot = base64.b64encode(buf.read()).decode('utf-8')

            # Générer le texte explicatif
            explain_text = '<h4>Impact des variables :</h4><ul>'
            for feat, val in top_features:
                impact_text = "augmente" if val > 0 else "diminue"
                explain_text += f'<li><b>{feat}</b> {impact_text} le risque de décrochage '
                explain_text += f'(impact: {"+" if val > 0 else ""}{val:.3f})</li>'
            explain_text += '</ul>'

            # Ajouter une explication générale
            explanation = {
                "title": "Comment interpréter ces résultats ?",
                "positive": "Une valeur positive (rouge) indique que cette variable augmente le risque de décrochage",
                "negative": "Une valeur négative (bleue) indique que cette variable diminue le risque de décrochage",
                "magnitude": "Plus la valeur est élevée en valeur absolue, plus l'influence de la variable est importante"
            }

        except Exception as e:
            print(f"Erreur lors de la génération de l'explicabilité: {str(e)}")
            explain_plot = None
            explain_text = "Explicabilité indisponible"
            explanation = None

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'explain_plot': explain_plot,
            'explain_text': explain_text,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier trouvé'}), 400
        file = request.files['file']
        df = pd.read_csv(file)
        predictions = model.predict(df)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[:, 1]
        else:
            proba = [None] * len(predictions)

        # Explicabilité SHAP pour chaque ligne
        try:
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor')
                classifier = model.named_steps.get('classifier')
            else:
                preprocessor = None
                classifier = model
            if preprocessor is not None:
                X_trans = preprocessor.transform(df)
            else:
                X_trans = df.values
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans)

            # Vérifier si shap_values est une liste (cas des modèles de classification binaire)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Prendre les valeurs pour la classe positive
        except Exception as e:
            print(f"Erreur lors du calcul SHAP: {str(e)}")
            shap_values = None

        results = []
        for i in range(len(df)):
            # Variables influentes (top 3)
            if shap_values is not None:
                try:
                    # Création d'une liste de tuples (nom_variable, valeur_shap)
                    feature_importance = list(zip(df.columns, shap_values[i]))
                    # Tri par valeur absolue des valeurs SHAP
                    top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:3]
                    # Formatage avec le nom de la variable et sa contribution (positive ou négative)
                    variables = ', '.join([f"{feat} ({'+'if val>0 else ''}{val:.3f})" for feat, val in top_features])
                except Exception as e:
                    print(f"Erreur lors du traitement des variables importantes: {str(e)}")
                    variables = 'Non disponible'
            else:
                variables = 'Non disponible'

            results.append({
                'etudiant': i+1,
                'prediction': int(predictions[i]),
                'probability': float(proba[i]) if proba[i] is not None else None,
                'variables': variables
            })

        # Ajout d'une explication pour l'interprétation des valeurs SHAP
        explanation = {
            "title": "Comment interpréter les variables influentes:",
            "positive": "Une valeur positive (+) indique que cette variable augmente le risque de décrochage",
            "negative": "Une valeur négative (-) indique que cette variable diminue le risque de décrochage",
            "magnitude": "Plus la valeur est grande en valeur absolue, plus son influence est importante"
        }

        return jsonify({
            'results': results,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

