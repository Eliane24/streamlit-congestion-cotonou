# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier


st.title("Smart City Cotonou : Prédiction de congestion routière")
st.markdown("Visualisation interactive de la congestion selon l'heure et la météo")

df = pd.read_csv("C:/Users/eaho/Documents/Pratices/dataset_congestion_cotonou.csv")

niveau_map = {"Faible": 0, "Moyen": 1, "Élevé": 2}
df["niveau_congestion_num"] = df["niveau_congestion"].map(niveau_map)

st.sidebar.header("Filtres interactifs")
jour = st.sidebar.selectbox("Jour de la semaine (0=Lundi)", sorted(df["jour_semaine"].unique()))

# ----------------------
# Graphique 1 : Répartition des niveaux de congestion par heure
# ----------------------
st.subheader(f"Répartition des niveaux de congestion par heure (Jour {jour})")

# Compter le nombre de Faible/Moyen/Élevé par heure
df_counts = df[df["jour_semaine"] == jour].groupby(["heure", "niveau_congestion"]).size().reset_index(name="count")

fig1 = px.bar(
    df_counts,
    x="heure",
    y="count",
    color="niveau_congestion",
    barmode="stack",
    labels={"count": "Nombre d'observations", "heure": "Heure", "niveau_congestion":"Niveau de congestion"},
    title=f"Congestion par heure pour le jour {jour}"
)
st.plotly_chart(fig1)

# ----------------------
# Graphique 2 : Importance des variables
# ----------------------
st.subheader("Importance des variables pour la prédiction")

X = df[["heure","jour_semaine","temp","rhum","prcp","wspd","pres"]]
y = df["niveau_congestion_num"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

df_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance")

fig2 = px.bar(
    df_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Importance des variables"
)
st.plotly_chart(fig2)

# ----------------------
# Graphique 3 : Congestion moyenne par heure et jour
# ----------------------
st.subheader("Congestion moyenne par heure et jour")

df_grouped = df.groupby(["jour_semaine", "heure"])["niveau_congestion_num"].mean().reset_index()

fig3 = px.line(
    df_grouped,
    x="heure",
    y="niveau_congestion_num",
    color="jour_semaine",
    labels={"niveau_congestion_num": "Congestion moyenne", "heure": "Heure"},
    title="Congestion moyenne par heure et jour de la semaine"
)
st.plotly_chart(fig3)

# ----------------------
# Prédiction interactive
# ----------------------
st.subheader("Prédiction interactive de congestion")

# Sidebar pour les variables météo et heure
heure = st.sidebar.slider("Heure", 0, 23, 8)
temp = st.sidebar.slider("Température (°C)", float(df["temp"].min()), float(df["temp"].max()), 30.0)
rhum = st.sidebar.slider("Humidité (%)", float(df["rhum"].min()), float(df["rhum"].max()), 80.0)
prcp = st.sidebar.slider("Précipitations (mm)", 0.0, float(df["prcp"].max()), 0.0)
wspd = st.sidebar.slider("Vitesse du vent (km/h)", 0.0, float(df["wspd"].max()), 10.0)
pres = st.sidebar.slider("Pression (hPa)", float(df["pres"].min()), float(df["pres"].max()), 1010.0)

X_input = pd.DataFrame({
    "heure": [heure],
    "jour_semaine": [jour],
    "temp": [temp],
    "rhum": [rhum],
    "prcp": [prcp],
    "wspd": [wspd],
    "pres": [pres]
})

pred_num = clf.predict(X_input)[0]

# Remapper le résultat en label
num_to_label = {0: "Faible", 1: "Moyen", 2: "Élevé"}
pred_label = num_to_label[pred_num]

# Affichage coloré
if pred_label == "Faible":
    st.success(f"⚡ Niveau de congestion prévu : {pred_label}")
elif pred_label == "Moyen":
    st.warning(f"⚠️ Niveau de congestion prévu : {pred_label}")
else:
    st.error(f"🚨 Niveau de congestion prévu : {pred_label}")

# ----------------------
# Export des données filtrées
# ----------------------
st.download_button(
    label="Télécharger les données filtrées",
    data=df[df["jour_semaine"]==jour].to_csv(index=False),
    file_name="predictions_congestion.csv",
    mime="text/csv"
)
