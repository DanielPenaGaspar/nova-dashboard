import streamlit as st
import pandas as pd
import principal as pr
import plotly.graph_objects as go

st.set_page_config(page_title="Predicción de precios", layout="wide")

# =====================================
# Botón para regresar al sitio web
# =====================================
col_nav, col_title = st.columns([1,6])

with col_nav:
    st.markdown(
        '<a href="https://novapredict.netlify.app/" target="_top">'
        '<button style="padding:10px 20px; font-size:16px;">Inicio</button>'
        '</a>',
        unsafe_allow_html=True
    )

with col_title:
    st.title("Predicción de precios de mercado")

# ============================
# Sidebar
# ============================
st.sidebar.header("Configuración")

opciones_activos = {
    "Bitcoin": 0,
    "Ethereum": 11,
    "Apple": 2,
    "Microsoft": 3,
    "Tesla": 4
}

activo = st.sidebar.selectbox("Activo", list(opciones_activos.keys()))
tipo_cambio = opciones_activos[activo]

days = st.sidebar.slider("Días a predecir", 1, 30, 7)

# ============================
# Ejecutar predicción
# ============================
if st.sidebar.button("Ejecutar predicción"):
    with st.spinner("Calculando predicción..."):
        df_real, df_pred, df_full = pr.run_pipeline(days, tipo_cambio)

        st.session_state["df_real"] = df_real
        st.session_state["df_pred"] = df_pred
        st.session_state["df_full"] = df_full

        if activo == "Ethereum":
            df_real_w, df_pred_w, df_full_w = pr.run_pipeline(days, 1)
            st.session_state["df_real_w"] = df_real_w
            st.session_state["df_pred_w"] = df_pred_w

# =========================================================
# PREDICCIÓN DIARIA
# =========================================================
if "df_real" in st.session_state:

    df_real = st.session_state["df_real"].copy()
    df_pred = st.session_state["df_pred"].copy()

    df_real.index = pd.to_datetime(df_real.index).tz_localize(None)
    df_pred.index = pd.to_datetime(df_pred.index).tz_localize(None)

    ultimo_real = df_real.index.max()
    df_pred = df_pred[df_pred.index > ultimo_real]

    historico_graf = df_real.tail(1)
    grafica = pd.concat([historico_graf, df_pred])

    st.subheader("Predicción diaria")

    fecha_min = grafica.index.min().to_pydatetime()
    fecha_max = grafica.index.max().to_pydatetime()

    inicio, fin = st.slider(
        "Rango de fechas (diario)",
        min_value=fecha_min,
        max_value=fecha_max,
        value=(fecha_min, fecha_max),
        format="YYYY-MM-DD"
    )

    grafica = grafica.loc[inicio:fin]

    df_plot = grafica.copy()
    df_plot["Tomorrow_Close"] = df_plot.get("Tomorrow_Close", 0).fillna(0)
    df_plot["Sugerencia"] = df_plot["Tomorrow_Close"].apply(
        lambda x: "🟢 Comprar" if x == 1 else "🔴 No comprar"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["Close"],
        mode="lines+markers",
        name="Precio predicho",
        customdata=df_plot[["Volume", "Sugerencia"]],
        hovertemplate="<b>Fecha:</b> %{x}<br><b>Close:</b> %{y:.2f}<br><b>Volumen:</b> %{customdata[0]}<br><b>Sugerencia:</b> %{customdata[1]}<extra></extra>"
    ))

    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Datos históricos")
        st.dataframe(df_real.tail(30))
    with col2:
        st.markdown("### Datos predichos")
        st.dataframe(df_pred)

# =========================================================
# ETHEREUM SEMANAL
# =========================================================
if activo == "Ethereum" and "df_real_w" in st.session_state:

    st.markdown("---")
    st.subheader("Predicción semanal Ethereum")

    df_real_w = st.session_state["df_real_w"].copy()
    df_pred_w = st.session_state["df_pred_w"].copy()

    df_real_w.index = pd.to_datetime(df_real_w.index).tz_localize(None)
    df_pred_w.index = pd.to_datetime(df_pred_w.index).tz_localize(None)

    ultimo_real_w = df_real_w.index.max()
    df_pred_w = df_pred_w[df_pred_w.index > ultimo_real_w]

    grafica_w = pd.concat([df_real_w.tail(1), df_pred_w])

    fecha_min = grafica_w.index.min().to_pydatetime()
    fecha_max = grafica_w.index.max().to_pydatetime()

    inicio, fin = st.slider(
        "Rango de fechas (semanal)",
        min_value=fecha_min,
        max_value=fecha_max,
        value=(fecha_min, fecha_max),
        format="YYYY-MM-DD",
        key="slider_semanal"
    )

    grafica_w = grafica_w.loc[inicio:fin]

    df_plot_w = grafica_w.copy()
    df_plot_w["NextWeek_Close"] = df_plot_w.get("NextWeek_Close", 0).fillna(0)
    df_plot_w["Sugerencia"] = df_plot_w["NextWeek_Close"].apply(
        lambda x: "🟢 Comprar" if x == 1 else "🔴 No comprar"
    )

    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(
        x=df_plot_w.index,
        y=df_plot_w["Close"],
        mode="lines+markers",
        name="Precio semanal predicho",
        customdata=df_plot_w[["Volume", "Sugerencia"]],
        hovertemplate="<b>Semana:</b> %{x}<br><b>Close:</b> %{y:.2f}<br><b>Volumen:</b> %{customdata[0]}<br><b>Sugerencia:</b> %{customdata[1]}<extra></extra>"
    ))

    fig_w.update_layout(legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig_w, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Datos históricos semanales")
        st.dataframe(df_real_w.tail(15))
    with col4:
        st.markdown("### Datos predichos semanales")
        st.dataframe(df_pred_w)

# =========================================================
# ANOMALÍAS
# =========================================================
if "df_full" in st.session_state:

    st.markdown("---")
    st.subheader("Detección de anomalías")

    df_anom = st.session_state["df_full"].copy()
    df_anom.index = pd.to_datetime(df_anom.index).tz_localize(None)

    total_anom = (df_anom["Anomalies"] == 1).sum()
    pct = total_anom / len(df_anom) * 100

    colA, colB = st.columns(2)
    colA.metric("Días anómalos", int(total_anom))
    colB.metric("% anomalías", f"{pct:.2f}%")

    fecha_min = df_anom.index.min().to_pydatetime()
    fecha_max = df_anom.index.max().to_pydatetime()

    inicio, fin = st.slider(
        "Rango de fechas (anomalías)",
        min_value=fecha_min,
        max_value=fecha_max,
        value=(fecha_min, fecha_max),
        format="YYYY-MM-DD",
        key="slider_anom"
    )

    df_anom = df_anom.loc[inicio:fin]
    df_points = df_anom[df_anom["Anomalies"] == 1]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_anom.index,
        y=df_anom["Close"],
        mode="lines",
        name="Precio"
    ))
    fig_price.add_trace(go.Scatter(
        x=df_points.index,
        y=df_points["Close"],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Día anómalo"
    ))

    fig_price.update_layout(legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig_price, use_container_width=True)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df_anom.index,
        y=df_anom["Volume"],
        name="Volumen"
    ))
    fig_vol.add_trace(go.Bar(
        x=df_points.index,
        y=df_points["Volume"],
        marker_color="red",
        name="Volumen anómalo"
    ))

    fig_vol.update_layout(legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig_vol, use_container_width=True)

    st.dataframe(df_points.tail(20))
