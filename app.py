import os
import pandas as pd
import numpy as np
import joblib
import io
import base64
from flask import Flask, request, render_template_string, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# === 1️⃣ CONFIGURATION & MODÈLE ===
app = Flask(__name__)
MODEL_PATH = "diabetes_model.pkl"

# Simulation d'un modèle pour l'exemple (à remplacer par ton .pkl si existant)
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Création d'un pipeline factice si le fichier est absent
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(n_estimators=50))
    ])
    # Entraînement rapide sur données aléatoires pour l'initialisation
    X_dummy = np.random.rand(10, 13)
    y_dummy = [0, 1] * 5
    pipe.fit(X_dummy, y_dummy)
    return pipe

model_pipeline = get_model()
COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_ratio', 
           'BMI2', 'Glucose_BMI', 'Glucose_Age', 'Pregnancies_Age']

# === 2️⃣ INTERFACE HTML/CSS (Template) ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse Médicale - Diabète</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 5px; }
        .author { text-align: center; color: #00bfa6; font-weight: bold; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .input-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-size: 0.9em; color: #666; }
        input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; }
        button { width: 100%; padding: 15px; background: #00bfa6; color: white; border: none; border-radius: 8px; font-size: 1.1em; cursor: pointer; margin-top: 20px; transition: 0.3s; }
        button:hover { background: #009688; }
        #result { margin-top: 30px; padding: 20px; border-radius: 8px; display: none; border: 1px solid #eee; }
        .result-positive { border-left: 10px solid #ff6b6b; background: #fff5f5; }
        .result-negative { border-left: 10px solid #00bfa6; background: #f0fff4; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analyse du Risque Diabète</h1>
        <div class="author">Développé par Leith Chqoubi</div>
        
        <form id="calcForm">
            <div class="grid">
                <div class="input-group"><label>Grossesses</label><input type="number" name="Pregnancies" value="2"></div>
                <div class="input-group"><label>Glycémie (mg/dL)</label><input type="number" name="Glucose" value="120"></div>
                <div class="input-group"><label>Pression (mmHg)</label><input type="number" name="BloodPressure" value="70"></div>
                <div class="input-group"><label>Épaisseur peau (mm)</label><input type="number" name="SkinThickness" value="25"></div>
                <div class="input-group"><label>Insuline (µU/mL)</label><input type="number" name="Insulin" value="80"></div>
                <div class="input-group"><label>IMC</label><input type="number" step="0.1" name="BMI" value="28.5"></div>
                <div class="input-group"><label>Hérédité (DPF)</label><input type="number" step="0.01" name="DiabetesPedigreeFunction" value="0.5"></div>
                <div class="input-group"><label>Âge</label><input type="number" name="Age" value="35"></div>
            </div>
            <button type="submit">Lancer l'analyse</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('calcForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });

            const res = await response.json();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = res.verdict === 'Positif' ? 'result-positive' : 'result-negative';
            
            resultDiv.innerHTML = `
                <h2 style="margin-top:0">Résultat : ${res.verdict}</h2>
                <p>Indice de probabilité : <strong>${res.probability}%</strong></p>
                <p>Niveau de risque : <strong>${res.risk}</strong></p>
            `;
        };
    </script>
</body>
</html>
"""

# === 3️⃣ ROUTES SERVEUR ===
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Conversion string -> float
    for key in data: data[key] = float(data[key])
    
    # Feature Engineering
    data["Glucose_Insulin_ratio"] = data["Glucose"] / max(data["Insulin"], 1)
    data["BMI2"] = data["BMI"] ** 2
    data["Glucose_BMI"] = data["Glucose"] * data["BMI"]
    data["Glucose_Age"] = data["Glucose"] / max(data["Age"], 1)
    data["Pregnancies_Age"] = data["Pregnancies"] * data["Age"]
    
    df = pd.DataFrame([data])[COLUMNS]
    
    # Prédiction
    prob = float(model_pipeline.predict_proba(df)[0, 1])
    verdict = "Positif" if prob >= 0.5 else "Négatif"
    risk = "Élevé" if prob > 0.7 else ("Modéré" if prob > 0.4 else "Faible")
    
    return jsonify({
        "probability": round(prob * 100, 2),
        "verdict": verdict,
        "risk": risk
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
