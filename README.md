# TelecomX — Modelado Predictivo de Evasión de Clientes

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Problema de Negocio](#2-problema-de-negocio)
3. [Dataset](#3-dataset)
4. [Pipeline de Machine Learning](#4-pipeline-de-machine-learning)
5. [Modelos Entrenados](#5-modelos-entrenados)
6. [Métricas de Evaluación](#6-métricas-de-evaluación)
7. [Hallazgos Clave](#7-hallazgos-clave)
8. [Recomendaciones Estratégicas](#8-recomendaciones-estratégicas)
9. [Estructura del Repositorio](#9-estructura-del-repositorio)
10. [Requisitos e Instalación](#10-requisitos-e-instalación)
11. [Uso](#11-uso)
12. [Tecnologías Utilizadas](#12-tecnologías-utilizadas)
13. [Contribuciones](#13-contribuciones)
14. [Licencia](#14-licencia)

---

## 1. Descripción del Proyecto

Este repositorio implementa el ciclo completo de **Machine Learning supervisado** para predecir la evasión de clientes (*Customer Churn*) de la empresa de telecomunicaciones ficticia **TelecomX**.

El proyecto forma parte del challenge de la **Especialización en Data Science — Alura LATAM (Grupo 9)** y continúa el trabajo de análisis exploratorio realizado previamente, tomando como punto de partida el dataframe limpio generado en esa fase.

---

## 2. Problema de Negocio

> **¿Qué clientes tienen alta probabilidad de abandonar el servicio en los próximos días?**

La evasión de clientes (*churn*) es uno de los problemas con mayor impacto económico en la industria de telecomunicaciones. Adquirir un nuevo cliente puede costar entre **5 y 25 veces más** que retener uno existente. Contar con un modelo predictivo permite:

- Activar campañas de retención **antes** de que el cliente cancele.
- Priorizar recursos comerciales hacia los perfiles de mayor riesgo.
- Diseñar productos y ofertas alineados con los factores de abandono identificados.

---

## 3. Dataset

| Atributo | Detalle |
|---|---|
| **Fuente** | [TelecomX_Data.json](https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json) |
| **Formato** | JSON anidado → normalizado con `pd.json_normalize` |
| **Registros** | ~7 000 clientes |
| **Variable objetivo** | `Evasor` (binaria: 0 = No evade, 1 = Evade) |
| **Desbalance de clases** | ~26 % evasores, ~74 % no evasores |

### Variables del Dataset

| Grupo | Variables |
|---|---|
| **Demográficas** | Género, Adulto_Mayor, Con_Pareja, Con_Dependientes |
| **Servicios contratados** | Servicio_Telefonico, Multiples_Líneas_Telefonicas, Servicio_Internet, Seguridad_Online, Copia_Seguridad_Online, Protección_Dispositivos, Soporte_Técnico, Servicio_Streaming_TV, Servicio_Streaming |
| **Cuenta** | Meses_como_cliente, Tipo_Contrato, Facturación_Sin_Papel, Método_Pago, Cargos_Mensuales, Cargos_Totales |
| **Ingeniería de features** | Número_Servicios (suma total de servicios activos) |

---

## 4. Pipeline de Machine Learning

```
Datos JSON (API)
      │
      ▼
Limpieza y Normalización
  • Renombrado de columnas
  • Conversión de tipos (bool, category, float)
  • Imputación de nulos (mediana en Cargos_Totales)
  • Feature engineering: Número_Servicios
      │
      ▼
Preprocesamiento para Modelado
  • Encoding: LabelEncoder (Género) + One-Hot (Tipo_Contrato, Método_Pago)
  • Escalado: StandardScaler en variables numéricas continuas
      │
      ▼
Análisis de Correlación
  • Mapa de calor Pearson
  • Selección: variables con |corr(Evasor)| ≥ 0.10
      │
      ▼
División Train / Test (80 % / 20 %, estratificada)
      │
      ▼
Entrenamiento de Modelos
  • Regresión Logística
  • Árbol de Decisión
  • Random Forest
      │
      ▼
Evaluación y Comparación
  • Accuracy, Precision, Recall, F1, ROC-AUC
  • Cross-validation estratificada (5-fold)
  • Matrices de confusión + Curvas ROC
      │
      ▼
Interpretación
  • Coeficientes LR + Feature Importances DT/RF
  • Ranking de variables por importancia promedio
```

---

## 5. Modelos Entrenados

| Modelo | Configuración |
|---|---|
| **Regresión Logística** | `max_iter=1000`, `class_weight='balanced'` |
| **Árbol de Decisión** | `max_depth=6`, `class_weight='balanced'` |
| **Random Forest** | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`, `n_jobs=-1` |

> Se usa `class_weight='balanced'` en todos los modelos para compensar el desbalance de clases sin recurrir a técnicas de re-muestreo (SMOTE/undersampling), manteniendo la reproducibilidad y simplificando la comparación.

---

## 6. Métricas de Evaluación

Se priorizan **Recall** y **ROC-AUC** sobre Accuracy, dado que el costo de un falso negativo (no detectar un evasor real) es mayor que el de un falso positivo en este contexto de negocio.

| Métrica | Justificación |
|---|---|
| **Accuracy** | Proporción global de predicciones correctas |
| **Precision** | De los predichos como evasores, cuántos lo son realmente |
| **Recall** | De los evasores reales, cuántos fueron detectados (**métrica principal**) |
| **F1-Score** | Media armónica entre Precision y Recall |
| **ROC-AUC** | Capacidad discriminativa del modelo a distintos umbrales |
| **CV F1 (5-fold)** | Estabilidad del modelo ante distintas particiones del conjunto de entrenamiento |

---

## 7. Hallazgos Clave

Los tres modelos coinciden en identificar los siguientes factores como los más determinantes para la predicción de evasión:

| Factor | Dirección | Interpretación |
|---|---|---|
| **Meses_como_cliente** | ↓ Evasión | Mayor antigüedad → mayor lealtad |
| **Tipo_Contrato (Mes a mes)** | ↑ Evasión | Principal predictor de abandono |
| **Cargos_Mensuales** | ↑ Evasión | Cargos elevados correlacionan con mayor propensión a irse |
| **Soporte_Técnico** | ↓ Evasión | Actúa como ancla de retención |
| **Seguridad_Online** | ↓ Evasión | Reduce la probabilidad de abandono |
| **Método_Pago (Cheque electrónico)** | ↑ Evasión | Método manual → mayor fricción y riesgo |
| **Con_Dependientes** | ↓ Evasión | Clientes con dependientes presentan mayor estabilidad |

---

## 8. Recomendaciones Estratégicas

1. **Migración de contratos mensuales**: Ofrecer incentivos (descuentos, beneficios adicionales) para migrar a planes anuales/bianuales, especialmente en los primeros 12–18 meses de vida del cliente.
2. **Programa de bienvenida intensivo**: Los clientes nuevos (< 18 meses) concentran el mayor riesgo. Un plan de acompañamiento temprano puede reducir significativamente la evasión.
3. **Paquetes con Soporte Técnico y Seguridad Online**: Incluirlos como anclas de retención, ofreciéndolos de forma gratuita en los primeros meses o en paquetes combinados.
4. **Eliminar la fricción del cheque electrónico**: Facilitar la migración a débito automático o pago con tarjeta mediante recordatorios y simplificación del proceso.
5. **Revisión de estructura de precios**: Diseñar propuestas de valor personalizadas para clientes de alto valor con cargos mensuales elevados.
6. **Sistema de alerta temprana**: Desplegar el modelo de Random Forest en producción para identificar proactivamente clientes con alta probabilidad de evasión y activar campañas de retención.

---

## 9. Estructura del Repositorio

```
TelecomX2G9/
├── telecomx_ml.ipynb        # Notebook principal: pipeline ML completo
├── README.md                # Documentación del proyecto
└── .github/
    └── instructions/        # Instrucciones de configuración del entorno
```

---

## 10. Requisitos e Instalación

### Prerrequisitos

- Python **3.9+**
- pip o conda

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Edbeto13/TelecomX2G9.git
cd TelecomX2G9

# (Opcional) Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn requests jupyter
```

### Dependencias principales

| Librería | Versión recomendada | Uso |
|---|---|---|
| `pandas` | ≥ 1.5 | Manipulación de datos |
| `numpy` | ≥ 1.23 | Operaciones numéricas |
| `matplotlib` | ≥ 3.6 | Visualización |
| `seaborn` | ≥ 0.12 | Visualización estadística |
| `scikit-learn` | ≥ 1.1 | Modelado y evaluación ML |
| `requests` | ≥ 2.28 | Descarga de datos vía HTTP |

---

## 11. Uso

### Ejecutar el notebook completo

```bash
jupyter notebook telecomx_ml.ipynb
```

O con JupyterLab:

```bash
jupyter lab
```

### Orden de ejecución de celdas

Ejecutar las celdas en orden secuencial (de arriba hacia abajo). Las dependencias entre celdas siguen el pipeline descrito en la [Sección 4](#4-pipeline-de-machine-learning).

> **Nota**: La celda de instalación de dependencias está comentada por defecto. Descomentarla sólo si es la primera ejecución en un entorno nuevo.

---

## 12. Tecnologías Utilizadas

| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.9+ |
| **Entorno** | Jupyter Notebook / JupyterLab |
| **Datos** | pandas, numpy, requests |
| **Visualización** | matplotlib, seaborn |
| **ML y evaluación** | scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, StandardScaler, LabelEncoder, métricas) |
| **Control de versiones** | Git / GitHub |