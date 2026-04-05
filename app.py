import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Fallas",
    page_icon="⚙️",
    layout="wide",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; }
h1 { color: #58a6ff; letter-spacing: 2px; }
h2 { color: #3fb950; font-size: 1.1rem; letter-spacing: 1px; margin-top: 2rem; }

.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.75rem;
    color: #8b949e;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.metric-box .value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'Share Tech Mono', monospace;
}

.result-ok {
    background: #0d2818;
    border: 2px solid #3fb950;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #3fb950;
    font-family: 'Share Tech Mono', monospace;
}
.result-fail {
    background: #2d0f0f;
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f85149;
    font-family: 'Share Tech Mono', monospace;
}
.prob-bar-bg {
    background: #21262d;
    border-radius: 6px;
    height: 14px;
    margin-top: 6px;
}
.stSlider > div > div { background: #21262d; }
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ── Carga y entrenamiento del modelo (cacheado) ──────────────────────────────
@st.cache_resource
def entrenar_modelos():
    df = pd.read_csv("ai_2020.csv")
    x = df.drop(['UDI', 'Product ID', 'Type', 'Machine failure',
                 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']

    # Split original
    x_train0, x_test0, y_train0, y_test0 = train_test_split(
        x, y, test_size=0.2, random_state=42)
    scaler0 = StandardScaler()
    x_tr0s = scaler0.fit_transform(x_train0)
    x_te0s = scaler0.transform(x_test0)
    rf0 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf0.fit(x_tr0s, y_train0)

    # SMOTE
    sm = SMOTE(k_neighbors=5, random_state=43)
    x_sm, y_sm = sm.fit_resample(x, y)
    y_sm = pd.Series(y_sm)

    x_train, x_test, y_train, y_test = train_test_split(
        x_sm, y_sm, test_size=0.2, random_state=44)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(x_train)
    X_te = scaler.transform(x_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_train)

    xgb = XGBClassifier(n_estimators=100, random_state=42,
                        use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_tr, y_train)

    return {
        "rf": rf, "xgb": xgb, "scaler": scaler,
        "rf0": rf0, "scaler0": scaler0,
        "x_test0": x_te0s, "y_test0": y_test0,
        "X_te": X_te, "y_test": y_test,
        "features": x.columns.tolist(),
        "df": df, "x": x, "y": y,
    }

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# ⚙️ DETECTOR DE FALLAS EN MÁQUINAS")
st.markdown("##### AI4I 2020 Predictive Maintenance Dataset")
st.divider()

with st.spinner("Entrenando modelos..."):
    m = entrenar_modelos()

# ── Sidebar: navegación ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ NAVEGACIÓN")
    seccion = st.radio("", [
        "🔍 Predictor",
        "📊 Métricas del Modelo",
        "📈 Curva ROC",
        "🌿 Importancia de Features",
        "⚖️ Comparación de Modelos",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Modelo activo**")
    modelo_elegido = st.selectbox("", ["Random Forest + SMOTE", "XGBoost + SMOTE"],
                                  label_visibility="collapsed")

# ── SECCIÓN: PREDICTOR ───────────────────────────────────────────────────────
if seccion == "🔍 Predictor":
    st.markdown("## 🔍 INGRESÁ LOS PARÁMETROS DE LA MÁQUINA")
    st.caption("Ajustá los valores del sensor y el modelo predice si hay riesgo de falla.")

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.slider("🌡️ Temperatura del Aire [K]", 295.0, 305.0, 300.0, 0.1)
        proc_temp = st.slider("🔥 Temperatura del Proceso [K]", 305.0, 315.0, 309.0, 0.1)
    with col2:
        rot_speed = st.slider("⚡ Velocidad Rotacional [rpm]", 1168, 2886, 1500)
        torque = st.slider("🔩 Torque [Nm]", 3.8, 76.6, 40.0, 0.1)

    tool_wear = st.slider("🪛 Desgaste de Herramienta [min]", 0, 253, 100)

    input_data = np.array([[air_temp, proc_temp, rot_speed, torque, tool_wear]])
    scaler = m["scaler"]
    input_scaled = scaler.transform(input_data)

    modelo = m["rf"] if modelo_elegido.startswith("Random") else m["xgb"]
    pred = modelo.predict(input_scaled)[0]
    prob = modelo.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    col_res, col_prob = st.columns([1, 1])

    with col_res:
        if pred == 0:
            st.markdown('<div class="result-ok">✅ SIN FALLA DETECTADA</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fail">⚠️ FALLA DETECTADA</div>',
                        unsafe_allow_html=True)

    with col_prob:
        st.markdown(f"**Probabilidad de falla: `{prob:.1%}`**")
        color = "#f85149" if prob > 0.5 else "#3fb950"
        st.markdown(f"""
        <div class="prob-bar-bg">
            <div style="width:{prob*100:.1f}%;background:{color};height:14px;border-radius:6px;transition:width 0.4s;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Umbral de decisión: 50%")

# ── SECCIÓN: MÉTRICAS ────────────────────────────────────────────────────────
elif seccion == "📊 Métricas del Modelo":
    st.markdown("## 📊 MÉTRICAS DEL MODELO")

    modelo = m["rf"] if modelo_elegido.startswith("Random") else m["xgb"]
    y_pred = modelo.predict(m["X_te"])
    y_test = m["y_test"]

    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip(
        [c1, c2, c3, c4],
        [("Accuracy", f"{acc:.1%}"),
         ("Precision", f"{rep['1']['precision']:.1%}"),
         ("Recall", f"{rep['1']['recall']:.1%}"),
         ("F1-Score", f"{rep['1']['f1-score']:.1%}")]
    ):
        col.markdown(f"""
        <div class="metric-box">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='#30363d',
                annot_kws={"color": "white", "size": 14})
    ax.set_xlabel('Predicho', color='#8b949e')
    ax.set_ylabel('Real', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    st.pyplot(fig)

# ── SECCIÓN: ROC ─────────────────────────────────────────────────────────────
elif seccion == "📈 Curva ROC":
    st.markdown("## 📈 CURVA ROC")
    st.caption("El AUC mide qué tan bien separa el modelo las clases. 1.0 = perfecto, 0.5 = aleatorio.")

    y_prob_rf0  = m["rf0"].predict_proba(m["x_test0"])[:, 1]
    y_prob_rf   = m["rf"].predict_proba(m["X_te"])[:, 1]
    y_prob_xgb  = m["xgb"].predict_proba(m["X_te"])[:, 1]

    fpr0, tpr0, _ = roc_curve(m["y_test0"], y_prob_rf0)
    fpr1, tpr1, _ = roc_curve(m["y_test"],  y_prob_rf)
    fpr2, tpr2, _ = roc_curve(m["y_test"],  y_prob_xgb)

    auc0 = roc_auc_score(m["y_test0"], y_prob_rf0)
    auc1 = roc_auc_score(m["y_test"],  y_prob_rf)
    auc2 = roc_auc_score(m["y_test"],  y_prob_xgb)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    ax.plot(fpr0, tpr0, label=f'RF sin SMOTE  (AUC={auc0:.3f})', color='#8b949e', lw=1.5, linestyle='--')
    ax.plot(fpr1, tpr1, label=f'RF con SMOTE  (AUC={auc1:.3f})', color='#58a6ff', lw=2)
    ax.plot(fpr2, tpr2, label=f'XGBoost SMOTE (AUC={auc2:.3f})', color='#3fb950', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aleatorio')
    ax.set_xlabel('Tasa de Falsos Positivos', color='#8b949e')
    ax.set_ylabel('Tasa de Verdaderos Positivos', color='#8b949e')
    ax.set_title('Curva ROC — Comparación de Modelos', color='#e6edf3', fontsize=12)
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(alpha=0.1, color='#8b949e')
    st.pyplot(fig)

# ── SECCIÓN: IMPORTANCIA DE FEATURES ────────────────────────────────────────
elif seccion == "🌿 Importancia de Features":
    st.markdown("## 🌿 IMPORTANCIA DE FEATURES")
    st.caption("Qué variables tienen más peso en la predicción del modelo.")

    modelo = m["rf"] if modelo_elegido.startswith("Random") else m["xgb"]
    importances = modelo.feature_importances_
    features = m["features"]
    idx = importances.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff']
    bars = ax.barh([features[i] for i in idx], importances[idx],
                   color=[colors[j % len(colors)] for j in range(len(idx))],
                   edgecolor='#30363d')
    ax.set_xlabel('Importancia', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(axis='x', alpha=0.1, color='#8b949e')
    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#e6edf3', fontsize=9)
    st.pyplot(fig)

# ── SECCIÓN: COMPARACIÓN ─────────────────────────────────────────────────────
elif seccion == "⚖️ Comparación de Modelos":
    st.markdown("## ⚖️ COMPARACIÓN DE MODELOS")

    y_pred_rf0  = m["rf0"].predict(m["x_test0"])
    y_pred_rf   = m["rf"].predict(m["X_te"])
    y_pred_xgb  = m["xgb"].predict(m["X_te"])
    y_prob_rf0  = m["rf0"].predict_proba(m["x_test0"])[:, 1]
    y_prob_rf   = m["rf"].predict_proba(m["X_te"])[:, 1]
    y_prob_xgb  = m["xgb"].predict_proba(m["X_te"])[:, 1]

    from sklearn.metrics import f1_score, precision_score, recall_score

    def row(nombre, y_true, y_pred, y_prob):
        return {
            "Modelo": nombre,
            "Accuracy":  f"{accuracy_score(y_true, y_pred):.4f}",
            "Precision": f"{precision_score(y_true, y_pred):.4f}",
            "Recall":    f"{recall_score(y_true, y_pred):.4f}",
            "F1-Score":  f"{f1_score(y_true, y_pred):.4f}",
            "AUC":       f"{roc_auc_score(y_true, y_prob):.4f}",
        }

    tabla = pd.DataFrame([
        row("RF sin SMOTE",  m["y_test0"], y_pred_rf0,  y_prob_rf0),
        row("RF con SMOTE",  m["y_test"],  y_pred_rf,   y_prob_rf),
        row("XGBoost SMOTE", m["y_test"],  y_pred_xgb,  y_prob_xgb),
    ])

    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.markdown("#### F1-Score por modelo")
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    nombres = tabla["Modelo"].tolist()
    f1s = [float(v) for v in tabla["F1-Score"]]
    bars = ax.bar(nombres, f1s, color=['#8b949e', '#58a6ff', '#3fb950'],
                  edgecolor='#30363d', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1-Score', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(axis='y', alpha=0.1, color='#8b949e')
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', color='#e6edf3', fontsize=10)
    st.pyplot(fig)
