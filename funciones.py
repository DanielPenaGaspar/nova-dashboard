import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest


# ==========================================================
# CARGAR DATOS==============================================
# ==========================================================
def load_data(symbol="BTC-USD", period="1y", interval="1d"):

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(symbol + ".csv")

    return df

# ==========================================================
# INDICADORES TÉCNICOS DIARIOS==============================
# ==========================================================
def add_indicators(df):

    df = df.copy()

    df["SMA7"] = df["Close"].rolling(7).mean()
    df["SMA21"] = df["Close"].rolling(21).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Weekly_Return"] = df["Close"].pct_change(5) * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_30d"] = df["Log_Returns"].rolling(30).std() * np.sqrt(365)

    df.fillna(0, inplace=True)

    return df


# ==========================================================
# INDICADORES SEMANALES (ETH)===============================
# ==========================================================
def add_indicators_weekly(df):

    df = df.copy()

    df["SMA7"] = df["Close"].rolling(1).mean()
    df["SMA21"] = df["Close"].rolling(3).mean()
    df["SMA50"] = df["Close"].rolling(7).mean()

    df["Weekly_Return"] = df["Close"].pct_change() * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=2, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=3, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=2, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_30d"] = df["Log_Returns"].rolling(3).std() * np.sqrt(365)

    df.fillna(0, inplace=True)

    return df


# ==========================================================
# CLASIFICACIÓN: SUBE O BAJA MAÑANA=========================
# ==========================================================
def datos_cl(df):

    df = df.copy()
    y = [1]

    for i in range(1, len(df)):
        if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]:
            y.append(1)
        else:
            y.append(0)

    df["Tomorrow_Close"] = y

    return df
# ==========================================================
# PREPARAR DATOS PARA CLASIFICACIÓN SEMANAL ETHEREUM========
# ==========================================================
def datos_clEt(df):

    df = df.copy()
    
    X_cl = df["Weekly_Return"].values

    y_cl = []
    ci = len(X_cl)

    for i in range(ci):

        if X_cl[i] > 0:
            y_cl.append(0)  # sube → no comprar
        else:
            y_cl.append(1)  # baja → comprar

    df["NextWeek_Close"] = y_cl
    
    return df

# ==========================================================
# CLASIFICACIÓN: BUENA O MALA COMPRA (%)====================
# ==========================================================
def datos_clBM(df, porcentaje):

    df = df.copy()

    y = [1, 1]

    for i in range(2, len(df)):
        actual = df["Close"].iloc[i]
        anterior = df["Close"].iloc[i - 2]

        incremento = ((actual - anterior) / anterior) * 100

        if incremento >= porcentaje:
            y.append(1)
        else:
            y.append(0)

    df["Target"] = y

    return df


# ==========================================================
# PREPARAR DATOS ML=========================================
# ==========================================================
def prepare_ml_data(df, col_cl):

    df = df.copy()

    df["High_next"] = df["High"].shift(-1)
    df["Low_next"] = df["Low"].shift(-1)
    df["Close_next"] = df["Close"].shift(-1)
    df["Volume_next"] = df["Volume"].shift(-1)

    df_train = df.iloc[:-1]

    X = df_train[["High", "Low", "Close", "Volume"]]
    y = df_train[["High_next", "Low_next", "Close_next", "Volume_next"]]
    y_c = df_train["Tomorrow_Close"]
    y_bm = df_train[col_cl]

    return df, X, y, y_c, y_bm
# ==========================================================
# PREPARAR DATOS PARA MODELO SEMANAL ETHEREUM===============
# ==========================================================
def prepare_ml_dataEt(df):

    df = df.copy()

    df["High_next"] = df["High"].shift(-1)
    df["Low_next"] = df["Low"].shift(-1)
    df["Close_next"] = df["Close"].shift(-1)
    df["Volume_next"] = df["Volume"].shift(-1)

    df_train = df.iloc[:-1]

    X = df_train[["High", "Low", "Close", "Volume"]].values
    y = df_train[["High_next", "Low_next", "Close_next", "Volume_next"]].values
    y_c = df_train["NextWeek_Close"].values

    return df, X, y, y_c


# ==========================================================
# MODELO REGRESIÓN==========================================
# ==========================================================
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        criterion="friedman_mse"
    )

    model.fit(X_train, y_train)

    return model


# ==========================================================
# MODELO XGBOOST CLASIFICACIÓN==============================
# ==========================================================
def train_modelXGBOOST(X, y):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_scaled, y)

    return model, scaler


# ==========================================================
# PREDICCIÓN FUTURA=========================================
# ==========================================================
def forecast_future(
    df_real,
    model,
    modelCL,
    scalerCL,
    modelCLBM,
    scalerCLBM,
    col_cl,
    days=7
):

    ultimo = df_real.iloc[-1]

    prev_close = ultimo["Close"]
    high = ultimo["High"]
    low = ultimo["Low"]
    close = ultimo["Close"]
    volume = ultimo["Volume"]

    fecha_inicial = df_real.index[-1]

    rows = []

    for i in range(days):

        X_pred = pd.DataFrame(
            [[high, low, close, volume]],
            columns=["High", "Low", "Close", "Volume"]
        )

        pred = model.predict(X_pred)[0]
        high_pred, low_pred, close_pred, volume_pred = pred

        fecha_nueva = fecha_inicial + pd.Timedelta(days=i+1)

        
        open_price = prev_close

        X_scaled = scalerCL.transform(X_pred)
        tc = modelCL.predict(X_scaled)[0]

        X_scaled_bm = scalerCLBM.transform(X_pred)
        bm = modelCLBM.predict(X_scaled_bm)[0]

        rows.append([
            fecha_nueva,
            open_price,
            high_pred,
            low_pred,
            close_pred,
            volume_pred,
            tc,
            bm
        ])

        prev_close = close_pred
        high = high_pred
        low = low_pred
        close = close_pred
        volume = volume_pred

    df_future = pd.DataFrame(
        rows,
        columns=[
            "Date", "Open", "High", "Low", "Close",
            "Volume", "Tomorrow_Close", col_cl
        ]
    )

    df_future.set_index("Date", inplace=True)

    return df_future


# ==========================================================
# ALTO VOLUMEN==============================================
# ==========================================================
# ==========================================================
# DETECCIÓN DE ALTO VOLUMEN=================================
# ==========================================================
def datos_AV(df):

    df = df.copy()

    volume = df["Volume"].values

    y = [1]

    for i in range(1, len(volume)):

        anterior = volume[i-1]
        actual = volume[i]

        if anterior != 0:
            incremento = (actual - anterior) / anterior * 100
        else:
            incremento = 0

        if incremento >= 70:
            y.append(1)
        else:
            y.append(0)

    df["Alto_volumen"] = y

    return df
# ==========================================================
# ISOLATION FOREST==========================================
# ==========================================================
def prepare_dataIF(df):

    return df[[
        "Daily_Return",
        "Volumen_Relativo_SMA7",
        "Volumen_Relativo_SMA21",
        "Volumen_Relativo_SMA50"
    ]].values


def train_modelIF(X):

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=76
    )

    model.fit(X)
    return model

# ==========================================================
# VOLUMEN RELATIVO PARA ISOLATION FOREST====================
# ==========================================================
def calcular_VR(df, sma):

    df = df.copy()

    volume = df["Volume"].values
    mean_volume = df[sma].values

    y = []

    for i in range(len(volume)):

        if mean_volume[i] != 0:
            relativo = volume[i] / mean_volume[i]
        else:
            relativo = volume[i]

        y.append(abs(relativo))

    col_name = "Volumen_Relativo_" + sma
    df[col_name] = y

    return df

# ==========================================================
# FORECAST SEMANAL ETHEREUM=================================
# ==========================================================
def forecast_futureEtW(df_real, model, modelCL, scalerCL, days=7):

    ultimo = df_real.iloc[-1]

    prev_close = ultimo["Close"]
    high = ultimo["High"]
    low = ultimo["Low"]
    close = ultimo["Close"]
    volume = ultimo["Volume"]

    fecha_inicial = df_real.index[-1]
    rows = []

    for i in range(days):

        X_pred = pd.DataFrame(
            [[high, low, close, volume]],
            columns=["High", "Low", "Close", "Volume"]
        )

        # regresión OHLCV
        pred = model.predict(X_pred)[0]
        high_pred, low_pred, close_pred, volume_pred = pred

        # clasificación semanal (CON SCALER)
        X_scaled = scalerCL.transform(X_pred)
        nw = modelCL.predict(X_scaled)[0]

        fecha_nueva = df_real.index[-1] + pd.DateOffset(weeks=i+1)
        open_price = prev_close

        rows.append([
            fecha_nueva,
            open_price,
            high_pred,
            low_pred,
            close_pred,
            volume_pred,
            nw
        ])

        # actualizar para siguiente iteración
        prev_close = close_pred
        high = high_pred
        low = low_pred
        close = close_pred
        volume = volume_pred

    df_future = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "NextWeek_Close"
        ]
    )

    df_future.set_index("Date", inplace=True)

    return df_future