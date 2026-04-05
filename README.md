# ⚙️ Detector de Fallas en Máquinas
### Machine Failure Detection — AI4I 2020 Predictive Maintenance Dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🔗 Demo en vivo
👉 **[Ver la app en Streamlit](https://detector-de-fallas-vweucfvtrsjosvp5lusfed.streamlit.app/)**

---

## 📌 Descripción

Proyecto de machine learning para predecir fallas en máquinas industriales a partir de datos de sensores en tiempo real. Se entrenaron y compararon múltiples modelos de clasificación, aplicando técnicas de balanceo de clases para mejorar la detección de fallas (clase minoritaria).

---

## 📊 Dataset

- **Fuente:** [AI4I 2020 Predictive Maintenance Dataset — UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Registros:** 10.000
- **Features:** Temperatura del aire, temperatura del proceso, velocidad rotacional, torque, desgaste de herramienta
- **Target:** `Machine failure` (0 = sin falla, 1 = falla)
- **Desbalance:** 96.6% sin falla / 3.4% con falla

---

## 🧠 Modelos entrenados

| Modelo | Técnica | AUC |
|---|---|---|
| Random Forest | Sin balanceo | ~0.98 |
| Random Forest | SMOTE | ~0.99 |
| XGBoost | SMOTE | ~0.99 |

> **SMOTE** (Synthetic Minority Oversampling Technique) se utilizó para balancear las clases y mejorar el recall de la clase minoritaria (fallas).

---

## 📈 Resultados — Mejor modelo (RF + SMOTE)

| Métrica | Valor |
|---|---|
| Accuracy | 94% |
| Precision | 94% |
| Recall | 95% |
| F1-Score | 94% |
| AUC | 0.99 |

---

## 🖥️ App interactiva

La app desarrollada en Streamlit permite:

- 🔍 **Predictor en tiempo real** — ingresás los parámetros del sensor y el modelo predice si hay falla, con probabilidad visual
- 📊 **Métricas del modelo** — accuracy, precision, recall, F1 y matriz de confusión
- 📈 **Curva ROC** — comparación de los 3 modelos
- 🌿 **Importancia de features** — qué variables tienen más peso en la predicción
- ⚖️ **Comparación de modelos** — tabla y gráfico comparativo

---

## 🗂️ Estructura del proyecto

```
detector-de-fallas/
│
├── app.py                # App principal de Streamlit
├── requirements.txt      # Dependencias del proyecto
├── ai_2020.csv           # Dataset
└── README.md             # Este archivo
```

---

## 🚀 Correr localmente

```bash
# Clonar el repositorio
git clone https://github.com/Nicfran/Detector-de-Fallas.git
cd detector-de-fallas

# Instalar dependencias
pip install -r requirements.txt

# Correr la app
streamlit run app.py
```

---

## 🛠️ Tecnologías

- **Python** — lenguaje principal
- **Pandas / NumPy** — procesamiento de datos
- **Scikit-learn** — modelos ML y métricas
- **XGBoost** — modelo de gradient boosting
- **Imbalanced-learn** — SMOTE para balanceo de clases
- **Matplotlib / Seaborn** — visualizaciones
- **Streamlit** — interfaz web interactiva

---

## 👤 Autor

Hecho con ❤️ como proyecto de portfolio personal.  
Si te gustó el proyecto, ¡dejá una ⭐ en el repositorio!
