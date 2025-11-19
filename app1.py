import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modèles de machine learning
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Modèles de séries temporelles
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from scipy import stats

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Anomalies & Prédiction des Pannes - Vallée d'Ulsim",
    page_icon="🔧",
    layout="wide"
)

def main():
    st.title("🔧 Système de Détection d'Anomalies et Prédiction des Pannes")
    st.markdown("**Projet : Maintenance Prédictive - Vallée d'Ulsim, Maroc**")
    st.markdown("---")
    
    # Introduction
    st.header("📋 Introduction")
    st.write("""
    Ce projet vise à développer un système intelligent de détection d'anomalies et de prédiction 
    des pannes pour les équipements industriels de la vallée d'Ulsim au Maroc. L'objectif principal 
    est de surveiller en temps réel le capteur de vibration et d'identifier les comportements 
    anormaux pouvant indiquer des défaillances potentielles.
    
    **Seuils de normalité définis par l'entreprise :**
    - Seuil minimum : 0.3
    - Seuil maximum : 7.0
    - Toute valeur en dehors de cette plage est considérée comme une anomalie
    """)
    
    # Chargement des données
    st.header("📊 Chargement et Exploration des Données")
    
    @st.cache_data
    def load_data():
        try:
            # Chargement du fichier avec le bon séparateur et gestion des décimales
            df = pd.read_csv(r"C:\Users\n\Downloads\Dataset_Cleaned22reel - Copy.csv", 
                           sep=';', 
                           decimal=',',
                           encoding='utf-8')
            
            # Conversion de la colonne Date en datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
            
            # Nettoyage des noms de colonnes (supprimer les espaces)
            df.columns = df.columns.str.strip()
            
            # Vérification des colonnes disponibles
            st.info(f"Colonnes disponibles : {list(df.columns)}")
            
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return None
    
    df = load_data()
    
    if df is not None:
        # Affichage des données
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        
        st.subheader("Informations sur le dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dimensions :**")
            st.write(f"- Lignes : {df.shape[0]}")
            st.write(f"- Colonnes : {df.shape[1]}")
            
        with col2:
            st.write("**Période couverte :**")
            st.write(f"- Début : {df['Date'].min()}")
            st.write(f"- Fin : {df['Date'].max()}")
        
        st.subheader("Statistiques descriptives - Vibration")
        st.dataframe(df['Vibration'].describe())
        
        # Vérification de la présence de valeurs manquantes
        st.subheader("Valeurs manquantes")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0])
        
        # Visualisations
        st.header("📈 Visualisations des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Série temporelle - Vibration")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Date'], df['Vibration'], linewidth=1, alpha=0.7)
            ax.axhline(y=7, color='r', linestyle='--', label='Seuil max (7.0)', linewidth=2)
            ax.axhline(y=0.3, color='g', linestyle='--', label='Seuil min (0.3)', linewidth=2)
            ax.set_ylabel('Vibration')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Distribution des vibrations")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Vibration'], bins=50, alpha=0.7, edgecolor='black', density=True)
            ax.axvline(x=7, color='r', linestyle='--', label='Seuil max', linewidth=2)
            ax.axvline(x=0.3, color='g', linestyle='--', label='Seuil min', linewidth=2)
            ax.set_xlabel('Vibration')
            ax.set_ylabel('Densité')
            ax.legend()
            st.pyplot(fig)
        
        # Analyse des corrélations
        st.subheader("Analyse des Corrélations")
        
        # Sélection des colonnes numériques pour la corrélation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Matrice de corrélation
            corr_matrix = df[numeric_cols].corr()
            
            # Focus sur la corrélation avec Vibration
            vibration_corr = corr_matrix['Vibration'].sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Corrélations avec Vibration :**")
                st.dataframe(vibration_corr)
            
            with col2:
                # Heatmap des corrélations (top 10 variables)
                top_vars = vibration_corr.head(11).index  # Inclut Vibration elle-même
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[top_vars].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                plt.title('Top des corrélations avec Vibration')
                st.pyplot(fig)
        
        # Détection d'anomalies basée sur les seuils
        st.header("🚨 Détection d'Anomalies par Seuils")
        
        # Application des seuils
        df['Anomalie_Seuil'] = (df['Vibration'] < 0.3) | (df['Vibration'] > 7.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistiques des anomalies par seuils")
            n_anomalies = df['Anomalie_Seuil'].sum()
            pourcentage = (n_anomalies / len(df)) * 100
            
            st.metric("Nombre d'anomalies détectées", n_anomalies)
            st.metric("Pourcentage d'anomalies", f"{pourcentage:.2f}%")
            
            # Répartition des types d'anomalies
            anomalies_basses = (df['Vibration'] < 0.3).sum()
            anomalies_hautes = (df['Vibration'] > 7.0).sum()
            
            st.write("**Répartition des anomalies :**")
            st.write(f"- Vibrations trop basses (< 0.3) : {anomalies_basses}")
            st.write(f"- Vibrations trop hautes (> 7.0) : {anomalies_hautes}")
        
        with col2:
            st.subheader("Visualisation des anomalies")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Données normales
            normal_data = df[~df['Anomalie_Seuil']]
            ax.scatter(normal_data['Date'], normal_data['Vibration'], 
                     c='blue', s=10, label='Normal', alpha=0.6)
            
            # Anomalies
            anomaly_data = df[df['Anomalie_Seuil']]
            ax.scatter(anomaly_data['Date'], anomaly_data['Vibration'], 
                     c='red', s=30, label='Anomalie', alpha=0.8)
            
            ax.axhline(y=7, color='r', linestyle='--', label='Seuil max (7.0)')
            ax.axhline(y=0.3, color='g', linestyle='--', label='Seuil min (0.3)')
            ax.set_ylabel('Vibration')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Détection d'anomalies avancée avec Machine Learning
        st.header("🤖 Détection d'Anomalies par Machine Learning")
        
        st.write("""
        ### Modèles de Détection d'Anomalies Utilisés :
        
        1. **Isolation Forest** : Idéal pour les données haute dimension, détecte les anomalies par isolation
        2. **One-Class SVM** : Apprend le pattern normal et identifie les déviations
        3. **Local Outlier Factor (LOF)** : Détecte les anomalies basées sur la densité locale
        4. **DBSCAN** : Clustering basé sur la densité, identifie les points isolés
        
        **Pourquoi ces modèles ?**
        - Isolation Forest : Excellente performance avec données industrielles
        - One-Class SVM : Robustesse au bruit
        - LOF : Détection d'anomalies locales
        - DBSCAN : Pas besoin de spécifier le nombre d'anomalies
        """)
        
        # Préparation des données pour ML
        features_for_anomaly = ['Vibration']
        
        # Ajout d'autres features si disponibles
        additional_features = ['Intensité', 'Vitesse', 'T.Bob1', 'T.Bob2', 'T.Bob3', 
                             'T.Pal1', 'T.Pal2', 'Débit_air']
        
        for feature in additional_features:
            if feature in df.columns:
                features_for_anomaly.append(feature)
        
        st.write(f"**Features utilisées :** {features_for_anomaly}")
        
        # Nettoyage des données
        X_anomaly = df[features_for_anomaly].dropna()
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_anomaly)
        
        # Application des modèles
        models = {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'One-Class SVM': OneClassSVM(nu=0.1),
            'Local Outlier Factor': LocalOutlierFactor(contamination=0.1),
        }
        
        anomalies_results = {}
        
        for name, model in models.items():
            if name == 'Local Outlier Factor':
                y_pred = model.fit_predict(X_scaled)
                y_pred = np.where(y_pred == -1, 1, 0)  # -1 pour anomalies dans LOF
            else:
                model.fit(X_scaled)
                y_pred = model.predict(X_scaled)
                y_pred = np.where(y_pred == -1, 1, 0)  # -1 pour anomalies
            
            anomalies_results[name] = y_pred
        
        # Comparaison des modèles
        st.subheader("Comparaison des Modèles de Détection")
        
        model_comparison = []
        for model_name, predictions in anomalies_results.items():
            n_anomalies = np.sum(predictions)
            percentage = (n_anomalies / len(predictions)) * 100
            model_comparison.append({
                'Modèle': model_name,
                'Anomalies détectées': n_anomalies,
                'Pourcentage': f"{percentage:.2f}%"
            })
        
        # Ajout de la méthode par seuils
        n_seuil_anomalies = df['Anomalie_Seuil'].sum()
        pourcentage_seuil = (n_seuil_anomalies / len(df)) * 100
        model_comparison.append({
            'Modèle': 'Méthode Seuils',
            'Anomalies détectées': n_seuil_anomalies,
            'Pourcentage': f"{pourcentage_seuil:.2f}%"
        })
        
        comparison_df = pd.DataFrame(model_comparison)
        st.dataframe(comparison_df)
        
        # Visualisation des résultats Isolation Forest (meilleur modèle généralement)
        st.subheader("Résultats détaillés - Isolation Forest")
        
        df_clean = df.dropna(subset=features_for_anomaly).copy()
        df_clean['Anomalie_IF'] = anomalies_results['Isolation Forest']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            normal_data_ml = df_clean[df_clean['Anomalie_IF'] == 0]
            anomaly_data_ml = df_clean[df_clean['Anomalie_IF'] == 1]
            
            ax.scatter(normal_data_ml['Date'], normal_data_ml['Vibration'], 
                     c='blue', s=10, label='Normal', alpha=0.6)
            ax.scatter(anomaly_data_ml['Date'], anomaly_data_ml['Vibration'], 
                     c='red', s=30, label='Anomalie ML')
            ax.axhline(y=7, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.7)
            ax.set_ylabel('Vibration')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Analyse des patterns temporels des anomalies
            anomalies_by_hour = df_clean[df_clean['Anomalie_IF'] == 1].copy()
            anomalies_by_hour['Hour'] = anomalies_by_hour['Date'].dt.hour
            
            fig, ax = plt.subplots(figsize=(10, 6))
            anomalies_by_hour['Hour'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_xlabel('Heure de la journée')
            ax.set_ylabel('Nombre d\'anomalies')
            ax.set_title('Distribution des anomalies par heure')
            st.pyplot(fig)
        
        # Analyse des séries temporelles
        st.header("📊 Analyse des Séries Temporelles")
        
        # Préparation des données pour l'analyse temporelle
        ts_data = df.set_index('Date')['Vibration']
        
        # Décomposition
        st.subheader("Décomposition de la Série Temporelle")
        
        try:
            # Utilisation d'une période raisonnable pour la décomposition
            period = min(24, len(ts_data) // 2)  # 24 heures ou moins si pas assez de données
            decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            decomposition.observed.plot(ax=axes[0], title='Série Originale')
            decomposition.trend.plot(ax=axes[1], title='Tendance')
            decomposition.seasonal.plot(ax=axes[2], title='Saisonnalité')
            decomposition.resid.plot(ax=axes[3], title='Résidus')
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Impossible de décomposer la série : {e}")
        
        # Tests statistiques
        st.subheader("Tests Statistiques pour la Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Test ADF (Stationnarité)
            st.write("**Test de Dickey-Fuller Augmenté (ADF)**")
            adf_result = adfuller(ts_data.dropna())
            st.write(f"Statistique ADF : {adf_result[0]:.4f}")
            st.write(f"p-value : {adf_result[1]:.4f}")
            
            if adf_result[1] <= 0.05:
                st.success("✅ La série est stationnaire (p-value ≤ 0.05)")
            else:
                st.warning("❌ La série n'est pas stationnaire (p-value > 0.05)")
        
        with col2:
            # Test de normalité Shapiro-Wilk
            st.write("**Test de Normalité Shapiro-Wilk**")
            shapiro_test = stats.shapiro(ts_data.dropna().sample(min(5000, len(ts_data))))
            st.write(f"Statistique : {shapiro_test[0]:.4f}")
            st.write(f"p-value : {shapiro_test[1]:.4f}")
            
            if shapiro_test[1] > 0.05:
                st.success("✅ La série suit une distribution normale")
            else:
                st.warning("❌ La série ne suit pas une distribution normale")
        
        # ACF et PACF
        st.subheader("Fonctions d'Autocorrélation")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        nlags = min(40, len(ts_data.dropna()) // 4)
        acf_vals = acf(ts_data.dropna(), nlags=nlags)
        pacf_vals = pacf(ts_data.dropna(), nlags=nlags)
        
        ax1.stem(range(len(acf_vals)), acf_vals)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axhline(y=1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.5, label='Intervalle de confiance 95%')
        ax1.axhline(y=-1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Fonction d\'Autocorrélation (ACF)')
        ax1.legend()
        
        ax2.stem(range(len(pacf_vals)), pacf_vals)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axhline(y=1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.5, label='Intervalle de confiance 95%')
        ax2.axhline(y=-1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Fonction d\'Autocorrélation Partielle (PACF)')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Modèles de prédiction
        st.header("🔮 Modèles de Prédiction")
        
        st.write("""
        ### Modèles de Prédiction Utilisés :
        
        1. **ARIMA** : Modèle linéaire classique pour séries temporelles
        2. **LSTM** : Réseau de neurones récurrent pour séquences temporelles
        
        **Pourquoi ces modèles ?**
        - ARIMA : Standard de l'industrie pour séries temporelles stationnaires
        - LSTM : Capture les dépendances complexes à long terme
        """)
        
        # Préparation des données pour la prédiction
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]
        
        # ARIMA
        st.subheader("Prédiction ARIMA")
        
        try:
            # Auto-ARIMA simplifié
            arima_model = ARIMA(train_data, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=len(test_data))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data.index, train_data, label='Entraînement', alpha=0.7)
            ax.plot(test_data.index, test_data, label='Test', alpha=0.7)
            ax.plot(test_data.index, arima_forecast, label='Prédiction ARIMA', linestyle='--')
            ax.set_ylabel('Vibration')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Calcul des métriques
            mse_arima = np.mean((test_data.values - arima_forecast)**2)
            rmse_arima = np.sqrt(mse_arima)
            mae_arima = np.mean(np.abs(test_data.values - arima_forecast))
            
            st.write("**Métriques ARIMA :**")
            st.write(f"- MSE : {mse_arima:.4f}")
            st.write(f"- RMSE : {rmse_arima:.4f}")
            st.write(f"- MAE : {mae_arima:.4f}")
            
        except Exception as e:
            st.error(f"Erreur dans ARIMA : {e}")
        
        # LSTM
        st.subheader("Prédiction LSTM")
        
        # Préparation des données pour LSTM
        scaler_lstm = MinMaxScaler()
        scaled_data = scaler_lstm.fit_transform(ts_data.values.reshape(-1, 1))
        
        # Création des séquences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        seq_length = 24
        X_seq, y_seq = create_sequences(scaled_data, seq_length)
        
        # Division train/test
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Reshape pour LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Construction du modèle LSTM
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Entraînement
        with st.spinner('Entraînement du modèle LSTM...'):
            history = lstm_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=20,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
        
        # Prédiction
        lstm_predictions = lstm_model.predict(X_test, verbose=0)
        lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions)
        y_test_actual = scaler_lstm.inverse_transform(y_test.reshape(-1, 1))
        
        # Visualisation LSTM
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Dates pour l'axe x
        test_dates = ts_data.index[train_size+seq_length:]
        
        ax.plot(test_dates, y_test_actual, label='Valeurs réelles', alpha=0.7)
        ax.plot(test_dates, lstm_predictions, label='Prédiction LSTM', linestyle='--')
        ax.set_ylabel('Vibration')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Métriques LSTM
        mse_lstm = np.mean((y_test_actual.flatten() - lstm_predictions.flatten())**2)
        rmse_lstm = np.sqrt(mse_lstm)
        mae_lstm = np.mean(np.abs(y_test_actual.flatten() - lstm_predictions.flatten()))
        
        st.write("**Métriques LSTM :**")
        st.write(f"- MSE : {mse_lstm:.4f}")
        st.write(f"- RMSE : {rmse_lstm:.4f}")
        st.write(f"- MAE : {mae_lstm:.4f}")
        
        # Comparaison finale des modèles
        st.header("📊 Comparaison Finale des Modèles")
        
        comparison_data = {
            'Modèle': ['ARIMA', 'LSTM'],
            'MSE': [mse_arima, mse_lstm],
            'RMSE': [rmse_arima, rmse_lstm],
            'MAE': [mae_arima, mae_lstm]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Section pour les prédictions futures
        st.header("🔮 Prédictions Futures et Alertes")
        
        # Sélection du nombre de pas à prédire
        n_steps = st.slider("Nombre d'heures à prédire", 1, 168, 24)
        
        if st.button("Générer les Prédictions Futures"):
            with st.spinner('Génération des prédictions...'):
                try:
                    # Utilisation du modèle LSTM pour la prédiction future
                    last_sequence = scaled_data[-seq_length:]
                    future_predictions = []
                    
                    current_sequence = last_sequence.reshape(1, seq_length, 1)
                    
                    for _ in range(n_steps):
                        next_pred = lstm_model.predict(current_sequence, verbose=0)
                        future_predictions.append(next_pred[0, 0])
                        
                        # Mise à jour de la séquence
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, 0] = next_pred[0, 0]
                    
                    future_predictions = scaler_lstm.inverse_transform(
                        np.array(future_predictions).reshape(-1, 1)
                    )
                    
                    # Création des dates futures
                    last_date = ts_data.index[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(hours=1),
                        periods=n_steps,
                        freq='H'
                    )
                    
                    # Visualisation
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Dernières 100 observations historiques
                    historical_dates = ts_data.index[-100:]
                    historical_values = ts_data.values[-100:]
                    
                    ax.plot(historical_dates, historical_values, 
                           label='Données Historiques', color='blue', linewidth=1)
                    ax.plot(future_dates, future_predictions, 
                           label='Prédictions Futures', color='red', linestyle='--', linewidth=2)
                    ax.axhline(y=7, color='r', linestyle='-', alpha=0.7, label='Seuil max')
                    ax.axhline(y=0.3, color='g', linestyle='-', alpha=0.7, label='Seuil min')
                    ax.set_ylabel('Vibration')
                    ax.set_xlabel('Date')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Alertes si dépassement de seuil
                    max_prediction = np.max(future_predictions)
                    min_prediction = np.min(future_predictions)
                    
                    st.subheader("🚨 Système d'Alerte")
                    
                    if max_prediction > 7:
                        st.error(f"**ALERTE CRITIQUE :** Vibration maximale prédite à {max_prediction:.2f} (dépasse le seuil de 7.0)")
                    elif min_prediction < 0.3:
                        st.error(f"**ALERTE CRITIQUE :** Vibration minimale prédite à {min_prediction:.2f} (en dessous du seuil de 0.3)")
                    else:
                        st.success("✅ Aucun risque de panne détecté dans les prédictions futures")
                    
                    # Détails des prédictions
                    st.write("**Détails des prédictions :**")
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Vibration_Prédite': future_predictions.flatten()
                    })
                    st.dataframe(pred_df)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération des prédictions : {e}")
        
        # Recommandations
        st.header("💡 Recommandations et Plan d'Action")
        
        st.write("""
        ### Conclusions et Recommandations :
        
        **1. Stratégie de Surveillance :**
        - Implémenter une surveillance en temps réel avec Isolation Forest
        - Maintenir la détection par seuils comme système de secours
        - Configurer des alertes automatiques pour les opérateurs
        
        **2. Maintenance Prédictive :**
        - Planifier les maintenances basées sur les prédictions LSTM
        - Surveiller particulièrement les périodes à haute risque identifiées
        - Maintenir un stock de pièces critiques basé sur les prédictions
        
        **3. Actions Immédiates :**
        - Vérifier les équipements correspondant aux anomalies détectées
        - Analyser les causes racines des vibrations anormales
        - Documenter les cas pour améliorer le modèle
        """)
        
        # Export des résultats
        st.header("📤 Export des Résultats")
        
        if st.button("Exporter les Résultats d'Analyse"):
            # Création d'un dataframe de résultats
            results_df = df.copy()
            if 'Anomalie_IF' in df_clean.columns:
                results_df = results_df.merge(df_clean[['Date', 'Anomalie_IF']], on='Date', how='left')
            
            # Sauvegarde
            results_df.to_csv('resultats_analyse_pannes.csv', index=False)
            st.success("Résultats exportés avec succès dans 'resultats_analyse_pannes.csv'")
    
    else:
        st.error("Impossible de charger les données. Vérifiez le chemin du fichier.")

if __name__ == "__main__":
    main()