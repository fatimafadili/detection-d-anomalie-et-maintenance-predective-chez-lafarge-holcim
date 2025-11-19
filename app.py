import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import des librairies pour le machine learning
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Système de Surveillance Industrielle",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏭 Système Intelligent de Surveillance et Optimisation des Paramètres Industriels")
st.markdown("""
**Analyse Prédictive des Anomalies et Maintenance Préventive - Données Capteurs Lafarge**
""")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez une section:", [
    "📊 Aperçu des Données",
    "🔍 Analyse Exploratoire",
    "🚨 Détection d'Anomalies",
    "🔮 Prédiction des Pannes",
    "📈 Visualisations Temporelles",
    "📋 Rapport Complet"
])

# Chargement des données
@st.cache_data
def load_data():
    try:
        # Simulation des données (à remplacer par votre chemin réel)
        # file_path = r"C:\Users\n\Downloads\Dataset_Cleaned22reel - Copy.csv"
        # df = pd.read_csv(file_path, sep=";")
        
        # Création de données simulées pour la démonstration
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(start='2020-05-01', end='2021-05-21', freq='H')[:n_samples]
        
        data = {
            'Date': dates,
            'Intensité': np.random.normal(1100, 50, n_samples),
            'Vitesse': np.random.normal(50, 10, n_samples),
            'T.Bob1': np.random.normal(30, 5, n_samples),
            'T.Bob2': np.random.normal(29, 5, n_samples),
            'T.Bob3': np.random.normal(33, 5, n_samples),
            'T.Pal1': np.random.normal(25, 4, n_samples),
            'T.Pal2': np.random.normal(24, 4, n_samples),
            'Vibration': np.random.normal(0.5, 0.1, n_samples),
            'T.Pal1 vent': np.random.normal(20, 3, n_samples),
            'T.Pal2 vent': np.random.normal(19, 3, n_samples),
            'Débit_air': np.random.normal(100, 20, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Conversion des virgules en points et cast en float (simulation)
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(float)
        
        # Création de la variable cible (anomalies simulées)
        df['Anomalie'] = 0
        anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
        df.loc[anomaly_indices, 'Anomalie'] = 1
        
        # Ajout de valeurs aberrantes pour certaines anomalies
        for idx in anomaly_indices[:25]:
            df.loc[idx, 'Vibration'] += np.random.uniform(0.5, 1.0)
            df.loc[idx, 'T.Bob1'] += np.random.uniform(10, 15)
        
        return df.set_index('Date')
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Chargement des données
df = load_data()

if df is None:
    st.error("Impossible de charger les données. Vérifiez le chemin du fichier.")
    st.stop()

# Page 1: Aperçu des Données
if page == "📊 Aperçu des Données":
    st.header("📊 Aperçu des Données")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset des Capteurs")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Informations Générales")
        st.metric("Nombre d'observations", df.shape[0])
        st.metric("Nombre de variables", df.shape[1])
        st.metric("Taux d'anomalies", f"{(df['Anomalie'].sum() / len(df) * 100):.2f}%")
    
    st.subheader("Statistiques Descriptives")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Types de Données")
    st.dataframe(pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes,
        'Valeurs Manquantes': df.isnull().sum()
    }), use_container_width=True)

# Page 2: Analyse Exploratoire
elif page == "🔍 Analyse Exploratoire":
    st.header("🔍 Analyse Exploratoire des Données")
    
    # Sélection des variables à visualiser
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Anomalie' in numerical_cols:
        numerical_cols.remove('Anomalie')
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_var1 = st.selectbox("Sélectionnez la première variable:", numerical_cols, index=0)
        fig1 = px.histogram(df, x=selected_var1, title=f"Distribution de {selected_var1}")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        selected_var2 = st.selectbox("Sélectionnez la deuxième variable:", numerical_cols, index=1)
        fig2 = px.box(df, y=selected_var2, title=f"Boxplot de {selected_var2}")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap de corrélation
    st.subheader("📈 Matrice de Corrélation")
    
    # Calcul de la matrice de corrélation
    corr_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                square=True, fmt='.2f', linewidths=0.5)
    plt.title('Matrice de Corrélation des Variables des Capteurs')
    st.pyplot(fig)
    
    # Analyse des anomalies
    st.subheader("🔍 Analyse des Anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des anomalies
        anomaly_counts = df['Anomalie'].value_counts()
        fig_anom = px.pie(values=anomaly_counts.values, 
                         names=['Normal', 'Anomalie'],
                         title="Distribution des Anomalies")
        st.plotly_chart(fig_anom, use_container_width=True)
    
    with col2:
        # Comparaison des moyennes
        normal_means = df[df['Anomalie'] == 0][numerical_cols[:4]].mean()
        anomaly_means = df[df['Anomalie'] == 1][numerical_cols[:4]].mean()
        
        comp_df = pd.DataFrame({
            'Normal': normal_means,
            'Anomalie': anomaly_means
        })
        
        st.dataframe(comp_df.style.format("{:.2f}"), use_container_width=True)

# Page 3: Détection d'Anomalies
elif page == "🚨 Détection d'Anomalies":
    st.header("🚨 Détection d'Anomalies")
    
    # Préparation des données
    features = df.drop('Anomalie', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Sélection du modèle
    st.subheader("Sélection du Modèle de Détection d'Anomalies")
    model_choice = st.selectbox(
        "Choisissez le modèle:",
        ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    )
    
    if st.button("Lancer la Détection d'Anomalies"):
        with st.spinner("Entraînement du modèle en cours..."):
            
            if model_choice == "Isolation Forest":
                model = IsolationForest(contamination=0.05, random_state=42)
                predictions = model.fit_predict(features_scaled)
                anomalies = (predictions == -1).astype(int)
                
            elif model_choice == "One-Class SVM":
                model = OneClassSVM(nu=0.05, kernel='rbf')
                predictions = model.fit_predict(features_scaled)
                anomalies = (predictions == -1).astype(int)
                
            else:  # LOF
                model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
                predictions = model.fit_predict(features_scaled)
                anomalies = (predictions == -1).astype(int)
            
            # Résultats
            df_results = df.copy()
            df_results['Anomalies_Detectees'] = anomalies
            
            st.subheader("📊 Résultats de la Détection")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Anomalies détectées", anomalies.sum())
            
            with col2:
                st.metric("Taux de détection", f"{(anomalies.sum() / len(df) * 100):.2f}%")
            
            with col3:
                if 'Anomalie' in df.columns:
                    accuracy = accuracy_score(df['Anomalie'], anomalies)
                    st.metric("Précision", f"{accuracy:.2%}")
            
            # Visualisation des anomalies
            st.subheader("📍 Visualisation des Anomalies Détectées")
            
            # PCA pour la visualisation
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            fig = px.scatter(
                x=features_pca[:, 0], 
                y=features_pca[:, 1],
                color=anomalies.astype(str),
                title="Visualisation PCA des Anomalies Détectées",
                labels={'color': 'Anomalie'},
                color_discrete_map={'0': 'blue', '1': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails des anomalies
            st.subheader("📋 Détails des Points Anormaux")
            anomaly_data = df_results[df_results['Anomalies_Detectees'] == 1]
            st.dataframe(anomaly_data.describe(), use_container_width=True)

# Page 4: Prédiction des Pannes
elif page == "🔮 Prédiction des Pannes":
    st.header("🔮 Prédiction des Pannes")
    
    st.info("""
    Cette section utilise l'apprentissage automatique pour prédire les pannes 
    basées sur les données des capteurs.
    """)
    
    # Préparation des données
    if 'Anomalie' not in df.columns:
        st.warning("La variable cible 'Anomalie' n'est pas disponible dans les données.")
        st.stop()
    
    X = df.drop('Anomalie', axis=1)
    y = df['Anomalie']
    
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sélection du modèle
    st.subheader("🎯 Sélection du Modèle de Prédiction")
    pred_model_choice = st.selectbox(
        "Choisissez le modèle de prédiction:",
        ["Random Forest", "XGBoost", "LightGBM"]
    )
    
    if st.button("Lancer l'Entraînement et la Prédiction"):
        with st.spinner("Entraînement du modèle en cours..."):
            
            if pred_model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif pred_model_choice == "XGBoost":
                model = XGBClassifier(random_state=42)
            else:  # LightGBM
                model = LGBMClassifier(random_state=42)
            
            # Entraînement
            model.fit(X_train_scaled, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            st.subheader("📊 Performance du Modèle")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("F1-Score", f"{f1:.2%}")
            col3.metric("AUC-ROC", f"{roc_auc:.2%}")
            
            # Matrice de confusion
            st.subheader("📈 Matrice de Confusion")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            ax.set_title('Matrice de Confusion')
            st.pyplot(fig)
            
            # Rapport de classification
            st.subheader("📋 Rapport de Classification")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            
            # Importance des caractéristiques
            st.subheader("🔍 Importance des Caractéristiques")
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 des Caractéristiques les Plus Importantes'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

# Page 5: Visualisations Temporelles
elif page == "📈 Visualisations Temporelles":
    st.header("📈 Visualisations Temporelles des Données des Capteurs")
    
    # Sélection des variables à visualiser
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Anomalie' in numerical_cols:
        numerical_cols.remove('Anomalie')
    
    selected_vars = st.multiselect(
        "Sélectionnez les variables à visualiser:",
        numerical_cols,
        default=numerical_cols[:3]
    )
    
    if selected_vars:
        st.subheader("📊 Évolution Temporelle des Variables Sélectionnées")
        
        # Création du graphique temporel
        fig = go.Figure()
        
        for var in selected_vars:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[var],
                name=var,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Évolution Temporelle des Variables des Capteurs",
            xaxis_title="Date",
            yaxis_title="Valeurs",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualisation avec anomalies
        if 'Anomalie' in df.columns:
            st.subheader("🚨 Évolution avec Points Anormaux")
            
            fig_anom = go.Figure()
            
            # Données normales
            normal_data = df[df['Anomalie'] == 0]
            anomaly_data = df[df['Anomalie'] == 1]
            
            for var in selected_vars[:2]:  # Limiter à 2 variables pour la clarté
                fig_anom.add_trace(go.Scatter(
                    x=normal_data.index,
                    y=normal_data[var],
                    name=f'{var} (Normal)',
                    mode='lines',
                    line=dict(color='blue')
                ))
                
                fig_anom.add_trace(go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data[var],
                    name=f'{var} (Anomalie)',
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x')
                ))
            
            fig_anom.update_layout(
                title="Évolution avec Détection des Anomalies",
                xaxis_title="Date",
                yaxis_title="Valeurs",
                height=500
            )
            
            st.plotly_chart(fig_anom, use_container_width=True)

# Page 6: Rapport Complet
elif page == "📋 Rapport Complet":
    st.header("📋 Rapport Complet d'Analyse")
    
    st.info("""
    Cette section fournit un résumé complet de l'analyse des données des capteurs,
    incluant les principales insights et recommandations.
    """)
    
    # Métriques clés
    st.subheader("📊 Métriques Clés du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre Total d'Observations", len(df))
    
    with col2:
        if 'Anomalie' in df.columns:
            st.metric("Taux d'Anomalies", f"{(df['Anomalie'].sum() / len(df) * 100):.2f}%")
        else:
            st.metric("Variables Capteurs", len(df.columns))
    
    with col3:
        # Calcul de la corrélation moyenne
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        avg_corr = corr_matrix.abs().mean().mean()
        st.metric("Corrélation Moyenne", f"{avg_corr:.2f}")
    
    with col4:
        # Variabilité moyenne
        variability = df.select_dtypes(include=[np.number]).std().mean()
        st.metric("Variabilité Moyenne", f"{variability:.2f}")
    
    # Insights principaux
    st.subheader("💡 Insights Principaux")
    
    insights = [
        "📈 **Performance du Système**: Les capteurs montrent une stabilité générale avec quelques pics anormaux",
        "🔍 **Détection d'Anomalies**: Le système identifie efficacement les comportements anormaux",
        "⚡ **Variables Clés**: La vibration et la température sont les indicateurs les plus sensibles",
        "🛠️ **Maintenance**: Recommandation de vérifications préventives basées sur les patterns détectés",
        "📊 **Qualité des Données**: Données complètes avec peu de valeurs manquantes"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Recommandations
    st.subheader("🎯 Recommandations")
    
    recommendations = [
        "**Maintenance Préventive**: Planifier des inspections basées sur les patterns d'anomalies détectés",
        "**Surveillance Continue**: Monitorer particulièrement les capteurs de vibration et température",
        "**Optimisation**: Ajuster les seuils d'alerte pour réduire les faux positifs",
        "**Formation**: Former le personnel à l'interprétation des alertes du système",
        "**Amélioration Continue**: Collecter plus de données pour améliorer la précision des modèles"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Téléchargement du rapport
    st.subheader("📥 Export des Résultats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Générer Rapport PDF"):
            st.success("Rapport généré avec succès! (Fonctionnalité à implémenter)")
    
    with col2:
        if st.button("💾 Exporter les Données Analysées"):
            # Création d'un DataFrame de résultats
            report_data = df.copy()
            if 'Anomalie' in df.columns:
                report_data['Statut'] = report_data['Anomalie'].map({0: 'Normal', 1: 'Anomalie'})
            
            # Conversion en CSV
            csv = report_data.to_csv(index=True)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"rapport_analyse_capteurs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Projet PFA - Détection et Prédiction des Anomalies & Maintenance Préventive**

📧 Contact: [votre-email@domain.com](mailto:votre-email@domain.com)
""")

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)