import pandas as pd
import funciones as fn

# ==========================================================
# UTILIDAD: LIMPIAR DATAFRAME===============================
# ==========================================================
def limpiar_df(df):
    df = df.copy()

    # índice datetime y único
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.duplicated(keep="first")]

    # eliminar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ==========================================================
# PIPELINE DIARIO===========================================
# ==========================================================
def Funciones_generales(df, days=30):

    df = limpiar_df(df)

    df = fn.add_indicators(df)
    df = fn.datos_cl(df)
    df = fn.datos_clBM(df, 3)

    col_cl = "Tomorrow_Close"

    df_ml, X, y, y_c, y_bm = fn.prepare_ml_data(df, col_cl)

    model = fn.train_model(X, y)
    modelCL, scalerCL = fn.train_modelXGBOOST(X, y_c)
    modelCLBM, scalerCLBM = fn.train_modelXGBOOST(X, y_bm)

    df_pred = fn.forecast_future(
        df_ml,
        model,
        modelCL,
        scalerCL,
        modelCLBM,
        scalerCLBM,
        col_cl,
        days
    )

    df_pred = limpiar_df(df_pred)

    # eliminar solapamientos de fechas
    df_pred = df_pred[~df_pred.index.isin(df.index)]

    # concat seguro
    df_full = pd.concat([df, df_pred], axis=0, ignore_index=False)

    df_full = limpiar_df(df_full)

    df_full = fn.datos_AV(df_full)
    df_full = fn.add_indicators(df_full)

    return df, df_pred, df_full


# ==========================================================
# PIPELINE SEMANAL ETH======================================
# ==========================================================
def Funciones_ETSem(df, days=7):

    df = limpiar_df(df)

    df = fn.add_indicators_weekly(df)
    df = fn.datos_clEt(df)

    df_ml, X, y, y_c = fn.prepare_ml_dataEt(df)

    model = fn.train_model(X, y)
    modelCL, scalerCL = fn.train_modelXGBOOST(X, y_c)

    df_pred = fn.forecast_futureEtW(
        df_ml,
        model,
        modelCL,
        scalerCL,
        days
    )
    """
    print("===== HISTÓRICO SEMANAL =====")
    print(df.tail())
    
    print("===== PREDICCIÓN SEMANAL =====")
    print(df_pred.head())
    """

    df_pred = limpiar_df(df_pred)
    df_pred = df_pred[~df_pred.index.isin(df.index)]

    df_full = pd.concat([df, df_pred], axis=0, ignore_index=False)
    df_full = limpiar_df(df_full)

    df_full = fn.add_indicators_weekly(df_full)

    return df, df_pred, df_full


# ==========================================================
# ANOMALÍAS=================================================
# ==========================================================
def Funciones_MoAnomalos(df):

    df = limpiar_df(df)

    df = fn.calcular_VR(df, "SMA7")
    df = fn.calcular_VR(df, "SMA21")
    df = fn.calcular_VR(df, "SMA50")

    X = fn.prepare_dataIF(df)
    modelIF = fn.train_modelIF(X)

    df["Anomalies"] = modelIF.predict(X)

    return df


# ==========================================================
# FUNCIÓN PRINCIPAL=========================================
# ==========================================================
def run_pipeline(days=30, tipo=0):

    match tipo:

        case 1:
            symbol = "ETH-USD"
            df = fn.load_data(symbol, period="2y", interval="1wk")
            df, df_pred, df_full = Funciones_ETSem(df, days)
            return df, df_pred, df_full

        case 2:
            symbol = "AAPL"
        case 3:
            symbol = "MSFT"
        case 4:
            symbol = "TSLA"
        case 5:
            symbol = "SPY"
        case 6:
            symbol = "QQQ"
        case 7:
            symbol = "^GSPC"
        case 8:
            symbol = "GC=F"
        case 9:
            symbol = "CL=F"
        case 11:
            symbol = "ETH-USD"
        case _:
            symbol = "BTC-USD"

    df = fn.load_data(symbol, period="1y", interval="1d")

    df, df_pred, df_full = Funciones_generales(df, days)
    df_full = Funciones_MoAnomalos(df_full)

    return df, df_pred, df_full


if __name__ == "__main__":
    df_real, df_pred, df_full = run_pipeline(30, 0)
    #print(df_full.tail())