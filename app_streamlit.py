import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

st.set_page_config(page_title="Análisis de entregas - DataCo", layout="wide")

st.title("Análisis descriptivo y predictivo de entregas")
st.caption("Dataset DataCo Supply Chain")

# -----------------------------
# CARGA DE DATOS
# -----------------------------
archivo = st.sidebar.file_uploader(
    "Sube la base de datos",
    type=["xlsm", "xlsx"]
)

if archivo is None:
    st.info("Sube el archivo DataCoSupplyChainDataset.xlsm para comenzar.")
    st.stop()

@st.cache_data
def cargar_datos(archivo):
    return pd.read_excel(archivo, engine="openpyxl")

df = cargar_datos(archivo)
data = df.copy()

# Variables derivadas usadas en el análisis

data["Brecha_dias"] = (
    data["Days for shipping (real)"]
    - data["Days for shipment (scheduled)"]
)

data["Entrega_a_tiempo"] = (
    data["Days for shipping (real)"]
    <= data["Days for shipment (scheduled)"]
).astype(int)

data["Cancelado"] = (data["Order Status"] == "CANCELED").astype(int)

# -----------------------------
# NAVEGACIÓN
# -----------------------------
seccion = st.sidebar.radio(
    "Sección",
    [
        "Resumen",
        "Análisis descriptivo",
        "Segmentaciones",
        "Modelo 1 - Entrega a tiempo",
        "Modelo 2 - Brecha de días",
        "Modelo 3 - Beneficio por pedido",
        "Modelo 4 - Tiempo de ciclo"
    ]
)

# -----------------------------
# RESUMEN
# -----------------------------
if seccion == "Resumen":
    st.subheader("Resumen de la base")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(data):,}")
    c2.metric("Columnas", data.shape[1])
    c3.metric("Entrega a tiempo", f"{data['Entrega_a_tiempo'].mean()*100:.2f}%")
    c4.metric("Tasa de cancelación", f"{data['Cancelado'].mean()*100:.2f}%")

    st.dataframe(data.head(10), use_container_width=True)

# -----------------------------
# ANÁLISIS DESCRIPTIVO
# -----------------------------
elif seccion == "Análisis descriptivo":
    st.subheader("Análisis descriptivo de variables")

    variables = [
        "Brecha_dias",
        "Benefit per order",
        "Days for shipping (real)",
        "Entrega_a_tiempo",
        "Cancelado"
    ]

    resultados = []
    for variable in variables:
        resultados.append({
            "Variable": variable,
            "Media": data[variable].mean(),
            "Mediana": data[variable].median(),
            "Moda": data[variable].mode().iloc[0],
            "Desv. Estándar": data[variable].std(),
            "Mínimo": data[variable].min(),
            "Máximo": data[variable].max()
        })

    tabla_descriptiva = pd.DataFrame(resultados)
    st.dataframe(tabla_descriptiva.round(2), use_container_width=True)

    st.markdown("### 1. Cumplimiento del tiempo de entrega")
    variable = "Entrega_a_tiempo"
    media = data[variable].mean()
    mediana = data[variable].median()
    moda = data[variable].mode().iloc[0]
    desv = data[variable].std()
    minimo = data[variable].min()
    maximo = data[variable].max()
    conteo = data[variable].value_counts().sort_index()
    tarde = conteo.get(0, 0)
    atiempo = conteo.get(1, 0)
    total = tarde + atiempo
    pct_tarde = tarde / total * 100
    pct_atiempo = atiempo / total * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.bar(
        ["Entrega tardía", "Entrega a tiempo"],
        [tarde, atiempo],
        color=["#E74C3C", "#2ECC71"]
    )
    for barra, cantidad, porcentaje in zip(
        barras, [tarde, atiempo], [pct_tarde, pct_atiempo]
    ):
        ax.text(
            barra.get_x() + barra.get_width()/2,
            barra.get_height()/2,
            f"{cantidad:,}\n({porcentaje:.2f}%)",
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold"
        )
    ax.set_title("Cumplimiento del tiempo de entrega")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Número de registros")
    texto = (
        f"Media: {media:.2f}\nMediana: {mediana:.2f}\nModa: {moda:.0f}\n"
        f"Desv. estándar: {desv:.2f}\nMínimo: {minimo:.0f}\nMáximo: {maximo:.0f}\n"
        f"Tasa a tiempo: {pct_atiempo:.2f}%"
    )
    ax.text(1.08, 0.95, texto, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)

    st.markdown("### 2. Brecha de días de envío")
    variable = "Brecha_dias"
    media = data[variable].mean()
    mediana = data[variable].median()
    moda = data[variable].mode().iloc[0]
    desv = data[variable].std()
    minimo = data[variable].min()
    maximo = data[variable].max()
    conteo = data[variable].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.bar(
        conteo.index.astype(str), conteo.values,
        width=0.85, color="#4A6FA5"
    )
    for barra, cantidad in zip(barras, conteo.values):
        ax.text(
            barra.get_x() + barra.get_width()/2,
            barra.get_height()/2,
            f"{cantidad:,}",
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold"
        )
    ax.set_title("Distribución de la brecha de días de envío")
    ax.set_xlabel("Brecha de días")
    ax.set_ylabel("Número de registros")
    texto = (
        f"Media: {media:.2f}\nMediana: {mediana:.2f}\nModa: {moda:.2f}\n"
        f"Desv. estándar: {desv:.2f}\nMínimo: {minimo:.2f}\nMáximo: {maximo:.2f}"
    )
    ax.text(1.08, 0.95, texto, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)

    st.markdown("### 3. Beneficio promedio por pedido")
    variable = "Benefit per order"
    media = data[variable].mean()
    mediana = data[variable].median()
    moda = data[variable].mode().iloc[0]
    desv = data[variable].std()
    minimo = data[variable].min()
    maximo = data[variable].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data[variable], bins=40, edgecolor="black", color="#E67E22")
    ax.axvline(media, linestyle="--", color="#C0392B", linewidth=2, label=f"Media: {media:.2f}")
    ax.axvline(mediana, linestyle="-.", color="#1E8449", linewidth=2, label=f"Mediana: {mediana:.2f}")
    ax.axvline(moda, linestyle=":", color="#5B2C6F", linewidth=2, label=f"Moda: {moda:.2f}")
    ax.set_title("Distribución del beneficio por pedido")
    ax.set_xlabel("Beneficio por pedido")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    texto = (
        f"Media: {media:.2f}\nMediana: {mediana:.2f}\nModa: {moda:.2f}\n"
        f"Desv. estándar: {desv:.2f}\nMínimo: {minimo:.2f}\nMáximo: {maximo:.2f}"
    )
    ax.text(1.08, 0.95, texto, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)

    st.markdown("### 4. Días reales de envío")
    variable = "Days for shipping (real)"
    media = data[variable].mean()
    mediana = data[variable].median()
    moda = data[variable].mode().iloc[0]
    desv = data[variable].std()
    minimo = data[variable].min()
    maximo = data[variable].max()
    conteo = data[variable].value_counts().sort_index()
    colores = ["#AED6F1", "#85C1E9", "#5DADE2", "#3498DB", "#2E86C1", "#2874A6", "#21618C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.bar(conteo.index.astype(str), conteo.values, color=colores[:len(conteo)])
    for barra, cantidad in zip(barras, conteo.values):
        ax.text(
            barra.get_x() + barra.get_width()/2,
            barra.get_height()/2,
            f"{cantidad:,}",
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold"
        )
    ax.set_title("Distribución de los días reales de envío")
    ax.set_xlabel("Días reales de envío")
    ax.set_ylabel("Número de registros")
    texto = (
        f"Media: {media:.2f}\nMediana: {mediana:.2f}\nModa: {moda:.2f}\n"
        f"Desv. estándar: {desv:.2f}\nMínimo: {minimo:.2f}\nMáximo: {maximo:.2f}"
    )
    ax.text(1.08, 0.95, texto, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)

    st.markdown("### 5. Cancelaciones")
    variable = "Cancelado"
    media = data[variable].mean()
    mediana = data[variable].median()
    moda = data[variable].mode().iloc[0]
    desv = data[variable].std()
    minimo = data[variable].min()
    maximo = data[variable].max()
    conteo = data[variable].value_counts().sort_index()
    no_cancelado = conteo.get(0, 0)
    cancelado = conteo.get(1, 0)
    total = no_cancelado + cancelado
    pct_no_cancelado = no_cancelado / total * 100
    pct_cancelado = cancelado / total * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.bar(["No cancelado", "Cancelado"], [no_cancelado, cancelado], color=["#2ECC71", "#E74C3C"])
    barra_no = barras[0]
    ax.text(
        barra_no.get_x() + barra_no.get_width()/2,
        barra_no.get_height()/2,
        f"{no_cancelado:,}\n({pct_no_cancelado:.2f}%)",
        ha="center", va="center", fontsize=11,
        color="white", fontweight="bold"
    )
    barra_cancelado = barras[1]
    ax.text(
        barra_cancelado.get_x() + barra_cancelado.get_width()/2,
        barra_cancelado.get_height() + total*0.01,
        f"{cancelado:,}\n({pct_cancelado:.2f}%)",
        ha="center", va="bottom", fontsize=11,
        color="black", fontweight="bold"
    )
    ax.set_title("Estado de cancelación de pedidos")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Número de registros")
    texto = (
        f"Media: {media:.2f}\nMediana: {mediana:.2f}\nModa: {moda:.0f}\n"
        f"Desv. estándar: {desv:.2f}\nMínimo: {minimo:.0f}\nMáximo: {maximo:.0f}\n"
        f"Tasa de cancelación: {pct_cancelado:.2f}%"
    )
    ax.text(1.08, 0.95, texto, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)

# -----------------------------
# SEGMENTACIONES
# -----------------------------
elif seccion == "Segmentaciones":
    st.subheader("Análisis segmentado")

    st.markdown("### Tasa de entrega a tiempo por modo de envío")
    tabla = data.groupby("Shipping Mode")["Entrega_a_tiempo"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(tabla.index, tabla.values, color="#5B8FF9")
    for barra, valor in zip(barras, tabla.values):
        ax.text(barra.get_x()+barra.get_width()/2, barra.get_height()/2, f"{valor:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Tasa de entrega a tiempo por modo de envío")
    ax.set_xlabel("Modo de envío")
    ax.set_ylabel("Entrega a tiempo (%)")
    st.pyplot(fig)

    st.markdown("### Brecha promedio de días por modo de envío")
    tabla = data.groupby("Shipping Mode")["Brecha_dias"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(tabla.index, tabla.values, color="#9B59B6")
    for barra, valor in zip(barras, tabla.values):
        ax.text(barra.get_x()+barra.get_width()/2, barra.get_height()/2, f"{valor:.2f}",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Brecha promedio de días por modo de envío")
    ax.set_xlabel("Modo de envío")
    ax.set_ylabel("Brecha promedio de días")
    st.pyplot(fig)

    st.markdown("### Tasa de entrega a tiempo por mercado")
    tabla = data.groupby("Market")["Entrega_a_tiempo"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(tabla.index, tabla.values, color="#16A085")
    for barra, valor in zip(barras, tabla.values):
        ax.text(barra.get_x()+barra.get_width()/2, barra.get_height()/2, f"{valor:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Tasa de entrega a tiempo por mercado")
    ax.set_xlabel("Mercado")
    ax.set_ylabel("Entrega a tiempo (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Beneficio promedio por estado de entrega")
    tabla = data.groupby("Delivery Status")["Benefit per order"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(tabla.index, tabla.values, color="#E67E22")
    for barra, valor in zip(barras, tabla.values):
        ax.text(barra.get_x()+barra.get_width()/2, barra.get_height()/2, f"{valor:.2f}",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Beneficio promedio por estado de entrega")
    ax.set_xlabel("Estado de entrega")
    ax.set_ylabel("Beneficio promedio")
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 10 categorías con menor tasa de entrega a tiempo")
    tabla = data.groupby("Category Name")["Entrega_a_tiempo"].mean().mul(100).sort_values().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.barh(tabla.index, tabla.values, color="#C0392B")
    for barra, valor in zip(barras, tabla.values):
        ax.text(barra.get_width()/2, barra.get_y()+barra.get_height()/2, f"{valor:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("10 categorías con menor tasa de entrega a tiempo")
    ax.set_xlabel("Entrega a tiempo (%)")
    ax.set_ylabel("Categoría")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Top 10 ciudades con mayor tasa de entrega tardía")
    tabla_ciudades = data.groupby("Order City")["Entrega_a_tiempo"].agg(["mean", "count"])
    tabla_ciudades = tabla_ciudades[tabla_ciudades["count"] >= 100]
    tabla_ciudades["Tasa_tardia"] = (1 - tabla_ciudades["mean"]) * 100
    tabla_ciudades = tabla_ciudades.sort_values("Tasa_tardia", ascending=False).head(10).sort_values("Tasa_tardia")
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.barh(tabla_ciudades.index, tabla_ciudades["Tasa_tardia"], color="#34495E")
    for barra, valor in zip(barras, tabla_ciudades["Tasa_tardia"]):
        ax.text(barra.get_width()/2, barra.get_y()+barra.get_height()/2, f"{valor:.1f}%",
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Top 10 ciudades de pedido con mayor tasa de entrega tardía")
    ax.set_xlabel("Tasa de entrega tardía (%)")
    ax.set_ylabel("Order City")
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# MODELO 1
# -----------------------------
elif seccion == "Modelo 1 - Entrega a tiempo":
    st.subheader("Modelo 1 - Random Forest Classifier")
    st.write(
        "Objetivo: clasificar cada pedido como entrega tardía (0) o entrega a tiempo (1)."
    )

    variables_predictoras = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Country",
        "Order City",
        "Category Name",
        "Customer Segment",
        "Type",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount"
    ]

    X = data[variables_predictoras]
    y = data["Entrega_a_tiempo"]

    categoricas = [
        "Shipping Mode", "Market", "Order Region", "Order Country",
        "Order City", "Category Name", "Customer Segment", "Type"
    ]
    numericas = [
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount"
    ]

    preprocesador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ("numericas", "passthrough", numericas)
        ]
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline_rf = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", modelo_rf)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    with st.spinner("Entrenando Random Forest..."):
        pipeline_rf.fit(X_train, y_train)
        y_pred = pipeline_rf.predict(X_test)
        y_prob = pipeline_rf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    st.markdown("### Métricas del modelo")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{precision:.2%}")
    c3.metric("Recall", f"{recall:.2%}")
    c4.metric("F1-score", f"{f1:.2%}")
    c5.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.markdown("### Reporte de clasificación")
    reporte = classification_report(
        y_test, y_pred,
        target_names=["Entrega tardía", "Entrega a tiempo"],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(reporte).T.round(3), use_container_width=True)

    st.markdown("### Matriz de confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Entrega tardía", "Entrega a tiempo"]
    )
    disp.plot(ax=ax_cm)
    ax_cm.set_title("Matriz de confusión - Random Forest")
    st.pyplot(fig_cm)

    st.markdown("### Curva ROC")
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
    ax_roc.set_title("Curva ROC - Predicción de entrega a tiempo")
    st.pyplot(fig_roc)

    st.markdown("### Variables con mayor influencia")
    encoder = pipeline_rf.named_steps["preprocesamiento"]
    nombres_variables = encoder.get_feature_names_out()
    importancias = pipeline_rf.named_steps["modelo"].feature_importances_

    tabla_importancia = pd.DataFrame({
        "Variable": nombres_variables,
        "Importancia": importancias
    })

    top_importancia = tabla_importancia.sort_values("Importancia", ascending=False).head(15)

    def limpiar_nombre(nombre):
        nombre = nombre.replace("numericas__", "")
        nombre = nombre.replace("categoricas__", "")
        nombre = nombre.replace("Days for shipment (scheduled)", "Días programados")
        nombre = nombre.replace("Order Item Discount", "Descuento del pedido")
        nombre = nombre.replace("Sales", "Ventas")
        nombre = nombre.replace("Order Item Quantity", "Cantidad del pedido")
        nombre = nombre.replace("Shipping Mode_", "")
        nombre = nombre.replace("Customer Segment_", "Segmento: ")
        nombre = nombre.replace("Type_", "Tipo: ")
        nombre = nombre.replace("Order City_", "Ciudad: ")
        return nombre

    top_importancia_limpia = top_importancia.copy()
    top_importancia_limpia["Variable"] = top_importancia_limpia["Variable"].apply(limpiar_nombre)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    ax_imp.barh(
        top_importancia_limpia["Variable"][::-1],
        top_importancia_limpia["Importancia"][::-1],
        color="#4A6FA5"
    )
    ax_imp.set_title("Variables con mayor influencia en la predicción de entrega a tiempo")
    ax_imp.set_xlabel("Importancia")
    ax_imp.set_ylabel("Variable")
    plt.tight_layout()
    st.pyplot(fig_imp)


# -----------------------------
# MODELO 2 - DESVIACIÓN / BRECHA DE DÍAS
# -----------------------------
elif seccion == "Modelo 2 - Brecha de días":
    st.subheader("Modelo 2 - ElasticNetCV")
    st.write(
        "Objetivo: predecir la brecha de días entre el tiempo real de envío y el tiempo programado."
    )

    variables_predictoras_d2 = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Category Name",
        "Customer Segment",
        "Type",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount"
    ]
    X_d2 = data[variables_predictoras_d2]
    y_d2 = data["Brecha_dias"]

    categoricas_d2 = [
        "Shipping Mode", "Market", "Order Region", "Category Name",
        "Customer Segment", "Type"
    ]
    numericas_d2 = [
        "Days for shipment (scheduled)", "Order Item Quantity",
        "Sales", "Order Item Discount"
    ]

    preprocesador_d2 = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas_d2),
            ("numericas", StandardScaler(), numericas_d2)
        ]
    )

    X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
        X_d2, y_d2, test_size=0.20, random_state=42
    )

    modelo_elasticnet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-3, 1, 50),
        cv=5,
        random_state=42,
        max_iter=10000,
        n_jobs=-1
    )

    pipeline_en = Pipeline(steps=[
        ("preprocesamiento", preprocesador_d2),
        ("modelo", modelo_elasticnet)
    ])

    with st.spinner("Entrenando ElasticNetCV..."):
        pipeline_en.fit(X_train_d2, y_train_d2)
        y_pred_d2 = pipeline_en.predict(X_test_d2)

    modelo_final_d2 = pipeline_en.named_steps["modelo"]
    mae_d2 = mean_absolute_error(y_test_d2, y_pred_d2)
    rmse_d2 = np.sqrt(mean_squared_error(y_test_d2, y_pred_d2))
    r2_d2 = r2_score(y_test_d2, y_pred_d2)

    st.markdown("### Métricas del modelo")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{mae_d2:.3f} días")
    c2.metric("RMSE", f"{rmse_d2:.3f} días")
    c3.metric("R²", f"{r2_d2:.4f}")
    c4.metric("Mejor alpha", f"{modelo_final_d2.alpha_:.4f}")
    c5.metric("Mejor l1_ratio", f"{modelo_final_d2.l1_ratio_:.2f}")

    st.markdown("### Valores reales vs. predichos")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_d2, y_pred_d2, alpha=0.3, color="#4A6FA5", edgecolors="k", s=20)
    ax.plot(
        [y_test_d2.min(), y_test_d2.max()],
        [y_test_d2.min(), y_test_d2.max()],
        "r--", lw=2, label="Ideal (y = x)"
    )
    ax.set_title("Valores reales vs predicciones - Brecha de días (ElasticNet)")
    ax.set_xlabel("Brecha real (días)")
    ax.set_ylabel("Brecha predicha (días)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Análisis de residuos")
    residuos_d2 = y_test_d2 - y_pred_d2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred_d2, residuos_d2, alpha=0.3, color="#E67E22", edgecolors="k", s=20)
    ax.axhline(y=0, color="red", linestyle="--", lw=2, label="Error Cero")
    ax.set_title("Análisis de residuos - ElasticNet (Dimensión 2)")
    ax.set_xlabel("Valor predicho de la brecha (días)")
    ax.set_ylabel("Residuo / Error (Real - Predicho)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Coeficientes y selección de variables")
    nombres_variables_d2 = pipeline_en.named_steps["preprocesamiento"].get_feature_names_out()
    coeficientes_d2 = modelo_final_d2.coef_

    def limpiar_nombre_d2(nombre):
        nombre = nombre.replace("categoricas__", "").replace("numericas__", "")
        nombre = nombre.replace("Days for shipment (scheduled)", "Días programados")
        nombre = nombre.replace("Order Item Discount", "Descuento del pedido")
        nombre = nombre.replace("Order Item Quantity", "Cantidad del pedido")
        nombre = nombre.replace("Sales", "Ventas")
        nombre = nombre.replace("Shipping Mode_", "Modo envío: ")
        nombre = nombre.replace("Market_", "Mercado: ")
        nombre = nombre.replace("Order Region_", "Región: ")
        nombre = nombre.replace("Category Name_", "Categoría: ")
        nombre = nombre.replace("Customer Segment_", "Segmento: ")
        nombre = nombre.replace("Type_", "Tipo: ")
        return nombre

    tabla_coef_d2 = pd.DataFrame({
        "Variable": [limpiar_nombre_d2(n) for n in nombres_variables_d2],
        "Coeficiente": coeficientes_d2
    })
    tabla_coef_d2["Coef_abs"] = tabla_coef_d2["Coeficiente"].abs()
    top_coef_d2 = tabla_coef_d2.sort_values("Coef_abs", ascending=False).head(15)
    n_ceros_d2 = (tabla_coef_d2["Coeficiente"] == 0).sum()
    n_total_d2 = len(tabla_coef_d2)

    colores_d2 = ["#2ECC71" if c > 0 else "#E74C3C" for c in top_coef_d2["Coeficiente"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_coef_d2["Variable"][::-1], top_coef_d2["Coeficiente"][::-1], color=colores_d2[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 15 coeficientes - ElasticNet (Brecha de días)")
    ax.set_xlabel("Coeficiente (efecto en días de brecha)")
    ax.set_ylabel("Variable")
    ax.text(
        0.98, 0.03,
        f"Variables llevadas a 0 por Lasso: {n_ceros_d2} de {n_total_d2}",
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        f"Tras la codificación se generaron {n_total_d2} variables; "
        f"{n_ceros_d2} fueron llevadas a cero por la regularización L1."
    )

# -----------------------------
# MODELO 3 - BENEFICIO POR PEDIDO
# -----------------------------
elif seccion == "Modelo 3 - Beneficio por pedido":
    st.subheader("Modelo 3 - XGBoost Regressor")
    st.write("Objetivo: predecir el beneficio por pedido (Benefit per order).")

    variables_predictoras_d3 = [
        "Shipping Mode",
        "Market",
        "Category Name",
        "Customer Segment",
        "Type",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount",
        "Order Item Profit Ratio"
    ]
    X_d3 = data[variables_predictoras_d3]
    y_d3 = data["Benefit per order"]

    categoricas_d3 = ["Shipping Mode", "Market", "Category Name", "Customer Segment", "Type"]
    numericas_d3 = [
        "Days for shipment (scheduled)", "Order Item Quantity", "Sales",
        "Order Item Discount", "Order Item Profit Ratio"
    ]

    preprocesador_d3 = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas_d3),
            ("numericas", StandardScaler(), numericas_d3)
        ]
    )

    X_train_d3, X_test_d3, y_train_d3, y_test_d3 = train_test_split(
        X_d3, y_d3, test_size=0.20, random_state=42
    )

    modelo_xgb = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    pipeline_xgb = Pipeline(steps=[
        ("preprocesamiento", preprocesador_d3),
        ("modelo", modelo_xgb)
    ])

    with st.spinner("Entrenando XGBoost Regressor..."):
        pipeline_xgb.fit(X_train_d3, y_train_d3)
        y_pred_d3 = pipeline_xgb.predict(X_test_d3)

    mae_d3 = mean_absolute_error(y_test_d3, y_pred_d3)
    rmse_d3 = np.sqrt(mean_squared_error(y_test_d3, y_pred_d3))
    r2_d3 = r2_score(y_test_d3, y_pred_d3)

    st.markdown("### Métricas del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"S/ {mae_d3:.2f}")
    c2.metric("RMSE", f"S/ {rmse_d3:.2f}")
    c3.metric("R²", f"{r2_d3:.4f}")

    st.markdown("### Valores reales vs. predichos")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_d3, y_pred_d3, alpha=0.3, color="#E67E22", edgecolors="k", s=20)
    ax.plot(
        [y_test_d3.min(), y_test_d3.max()],
        [y_test_d3.min(), y_test_d3.max()],
        "r--", lw=2, label="Ideal (y = x)"
    )
    ax.set_title("Valores reales vs predicciones - Beneficio por pedido (XGBoost)")
    ax.set_xlabel("Valor real (S/)")
    ax.set_ylabel("Valor predicho (S/)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Importancia de variables")
    encoder_d3 = pipeline_xgb.named_steps["preprocesamiento"]
    nombres_cat_d3 = list(
        encoder_d3.named_transformers_["categoricas"].get_feature_names_out(categoricas_d3)
    )
    nombres_variables_d3 = nombres_cat_d3 + numericas_d3
    importancias_d3 = pipeline_xgb.named_steps["modelo"].feature_importances_

    tabla_importancia_d3 = pd.DataFrame({
        "Variable": nombres_variables_d3,
        "Importancia": importancias_d3
    }).sort_values("Importancia", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        tabla_importancia_d3["Variable"][::-1],
        tabla_importancia_d3["Importancia"][::-1],
        color="#E67E22"
    )
    ax.set_title("Top 15 variables más importantes - Beneficio por pedido")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Análisis de residuos")
    residuos_d3 = y_test_d3 - y_pred_d3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred_d3, residuos_d3, alpha=0.3, color="#E67E22", edgecolors="k", s=20)
    ax.axhline(y=0, color="red", linestyle="--", lw=2, label="Error Cero")
    ax.set_title("Análisis de residuos - XGBoost (Dimensión 3)")
    ax.set_xlabel("Valor predicho del beneficio (S/)")
    ax.set_ylabel("Residuo / Error (Real - Predicho)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# MODELO 4 - EFICIENCIA / TIEMPO DE CICLO
# -----------------------------
elif seccion == "Modelo 4 - Tiempo de ciclo":
    st.subheader("Modelo 4 - Gradient Boosting Regressor")
    st.write("Objetivo: predecir el tiempo del ciclo del pedido, medido en días.")

    data_d4 = data.copy()
    data_d4["Order_Date"] = pd.to_datetime(data_d4["order date (DateOrders)"])
    data_d4["Shipping_Date"] = pd.to_datetime(data_d4["shipping date (DateOrders)"])
    data_d4["Cycle_Time"] = (data_d4["Shipping_Date"] - data_d4["Order_Date"]).dt.days

    variables_predictoras_d4 = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Category Name",
        "Customer Segment",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount"
    ]
    X_d4 = data_d4[variables_predictoras_d4]
    y_d4 = data_d4["Cycle_Time"]

    variables_categoricas_d4 = [
        "Shipping Mode", "Market", "Order Region", "Category Name", "Customer Segment"
    ]

    # Se conserva el preprocesamiento tal como fue entregado en el notebook de la dimensión 4:
    # el ColumnTransformer codifica las variables categóricas.
    preprocesador_d4 = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_categoricas_d4)
        ]
    )

    X_train_d4, X_test_d4, y_train_d4, y_test_d4 = train_test_split(
        X_d4, y_d4, test_size=0.20, random_state=42
    )

    modelo_gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipeline_gbr = Pipeline(steps=[
        ("preprocesamiento", preprocesador_d4),
        ("modelo", modelo_gbr)
    ])

    with st.spinner("Entrenando Gradient Boosting Regressor..."):
        pipeline_gbr.fit(X_train_d4, y_train_d4)
        y_pred_d4 = pipeline_gbr.predict(X_test_d4)

    mae_d4 = mean_absolute_error(y_test_d4, y_pred_d4)
    rmse_d4 = np.sqrt(mean_squared_error(y_test_d4, y_pred_d4))
    r2_d4 = r2_score(y_test_d4, y_pred_d4)

    st.markdown("### Métricas del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae_d4:.3f} días")
    c2.metric("RMSE", f"{rmse_d4:.3f} días")
    c3.metric("R²", f"{r2_d4:.3f}")

    st.markdown("### Valores reales vs. predichos")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test_d4, y_pred_d4, alpha=0.3)
    ax.plot(
        [y_test_d4.min(), y_test_d4.max()],
        [y_test_d4.min(), y_test_d4.max()],
        "--"
    )
    ax.set_xlabel("Tiempo real del ciclo (días)")
    ax.set_ylabel("Tiempo predicho (días)")
    ax.set_title("Gradient Boosting Regressor: Valores reales vs predicciones")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Análisis de residuos")
    residuos_d4 = y_test_d4 - y_pred_d4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_d4, residuos_d4, alpha=0.3)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Valores predichos")
    ax.set_ylabel("Residuos")
    ax.set_title("Análisis de residuos - Gradient Boosting Regressor")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Importancia de variables")
    nombres_variables_d4 = pipeline_gbr.named_steps["preprocesamiento"].get_feature_names_out()
    importancias_d4 = pipeline_gbr.named_steps["modelo"].feature_importances_
    df_importancia_d4 = pd.DataFrame({
        "Variable": nombres_variables_d4,
        "Importancia": importancias_d4
    }).sort_values(by="Importancia", ascending=False)
    top15_d4 = df_importancia_d4.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top15_d4["Variable"], top15_d4["Importancia"])
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variables")
    ax.set_title("Importancia de variables - Gradient Boosting Regressor")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
