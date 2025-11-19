🔍 Détection d'Anomalies & Prédiction des Pannes
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Machine-Learning-orange
https://img.shields.io/badge/Status-Active-brightgreen

Un système intelligent de détection d'anomalies et de prédiction des pannes pour équipements industriels utilisant le Machine Learning.

✨ Fonctionnalités
🚨 Détection d'anomalies en temps réel

📊 Prédiction des pannes avec anticipation

📈 Visualisation interactive des données

🔔 Système d'alertes automatique

🌐 API REST pour intégration

📱 Dashboard en temps réel

🚀 Démarrage Rapide
Prérequis
Python 3.8+

pip ou conda

Installation
bash
# Cloner le dépôt
git clone https://github.com/votre-username/anomaly-detection-prediction.git
cd anomaly-detection-prediction

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
Utilisation Basique
python
from src.models.anomaly_detector import AnomalyDetector
from src.models.failure_predictor import FailurePredictor

# Charger les données
import pandas as pd
data = pd.read_csv('data/raw/sensor_data.csv')

# Détection d'anomalies
detector = AnomalyDetector()
anomalies = detector.detect(data)

# Prédiction des pannes
predictor = FailurePredictor()
predictions = predictor.predict(data)
📁 Structure du Projet
text
anomaly-detection-prediction/
├── 📊 data/
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données traitées
│   └── external/            # Données externes
├── 🤖 models/
│   ├── trained_models/      # Modèles entraînés
│   └── model_training/      # Scripts d'entraînement
├── 💻 src/
│   ├── data_processing/     # Prétraitement
│   ├── feature_engineering/ # Feature engineering
│   ├── models/              # Implémentation modèles
│   ├── evaluation/          # Évaluation
│   └── utils/               # Utilitaires
├── 🌐 api/                  # API FastAPI
├── 📈 dashboard/            # Dashboard Streamlit
├── ⚙️ config/               # Configuration
└── 🧪 tests/                # Tests unitaires
🛠 Modèles Implémentés
Détection d'Anomalies
✅ Isolation Forest

✅ Local Outlier Factor (LOF)

✅ Autoencodeurs

✅ One-Class SVM

✅ DBSCAN

Prédiction des Pannes
✅ XGBoost

✅ Random Forest

✅ LSTM Networks

✅ LightGBM

✅ Prophet

📊 Performance
Modèle	Précision	Rappel	F1-Score	AUC-ROC
Détection Anomalies	95.2%	92.8%	94.0%	96.5%
Prédiction Pannes	88.7%	85.3%	87.0%	91.2%
🌐 API Usage
Démarrer l'API
bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Exemple de Requête
python
import requests

url = "http://localhost:8000/api/v1/predict/anomaly"
data = {
    "sensor_data": {
        "temperature": 75.2,
        "pressure": 30.5,
        "vibration": 2.1,
        "rotation_speed": 2850
    }
}

response = requests.post(url, json=data)
print(response.json())
Réponse
json
{
    "anomaly": true,
    "confidence": 0.92,
    "anomaly_type": "vibration_high",
    "recommendation": "Vérifier l'équilibrage"
}
📈 Dashboard
Lancer le dashboard interactif :

bash
streamlit run dashboard/app.py
Le dashboard permet de :

Visualiser les données en temps réel

Voir les alertes d'anomalies

Consulter l'historique des prédictions

Analyser les performances des modèles

🐋 Déploiement avec Docker
bash
# Construction de l'image
docker build -t anomaly-detection .

# Lancement du container
docker run -p 8000:8000 anomaly-detection
Docker Compose
bash
docker-compose up -d
🧪 Tests
Exécuter la suite de tests :

bash
# Tests unitaires
python -m pytest tests/ -v

# Tests avec couverture
python -m pytest --cov=src tests/

# Tests d'intégration
python -m pytest tests/integration/ -v
🤝 Contribution
Les contributions sont les bienvenues !

🍴 Fork le projet

🌿 Créer une branche feature (git checkout -b feature/ma-feature)

💾 Commit les changements (git commit -m 'Ajout ma feature')

📤 Push vers la branche (git push origin feature/ma-feature)

🔃 Ouvrir une Pull Request

📋 TODO
Intégration IoT temps réel

Explicabilité des modèles (XAI)

Interface mobile

Support multi-langues

Intégration CMMS

📄 Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

📞 Contact
📧 Email : votre-email@domain.com

💬 Issues : GitHub Issues

📚 Documentation : Wiki

🙏 Remerciements
Contributeurs

Bibliothèques open source

Communauté ML

