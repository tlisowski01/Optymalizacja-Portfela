import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📈 AI Optymalizacja Portfela - Spółki WIG20")

# Lista spółek WIG20 z tickerami na Yahoo Finance (odpowiedniki dla Stooq nie działają z yfinance)
wig20_tickers = {
    "PKN Orlen": "PKN.OL",
    "PKO BP": "PKO.WA",
    "PZU": "PZU.WA",
    "KGHM": "KGH.WA",
    "Santander Bank Polska": "SPL.WA",
    "LPP": "LPP.WA",
    "CD Projekt": "CDR.WA",
    "Dino": "DNP.WA",
    "Allegro": "ALE.WA",
    "CCC": "CCC.WA",
    "Cyfrowy Polsat": "CPS.WA",
    "JSW": "JSW.WA",
    "mBank": "MBK.WA",
    "Orange Polska": "OPL.WA",
    "Pepco": "PCO.WA",
    "Grupa Kęty": "KTY.WA",
    "Alior Bank": "ALR.WA",
    "Asseco Poland": "ACP.WA",
    "Tauron": "TPE.WA",
    "PGE": "PGE.WA"
}

# Użytkownik wybiera spółki
selected_stocks = st.multiselect("Wybierz spółki z WIG20:", list(wig20_tickers.keys()))

# Zakres dat
end_date = datetime.today()
start_date = end_date - timedelta(days=3 * 365)

# Pobieranie danych
@st.cache_data
def download_data(tickers):
    data = {}
    for name in tickers:
        ticker = wig20_tickers[name]
        df = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
        df.name = name
        data[name] = df
    return pd.concat(data.values(), axis=1)

if selected_stocks:
    st.info("Pobieranie danych...")
    prices = download_data(selected_stocks)
    st.success("Dane pobrane!")

    # Wyświetlenie wykresu cen
    st.line_chart(prices)

    # Obliczanie dziennych zwrotów
    returns = prices.pct_change().dropna()

    st.subheader("🔍 Optymalizacja portfela (AI)")

    # Przygotowanie danych do AI
    X = returns.copy()
    y = returns.mean(axis=1)  # sztuczna metryka do regresji

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    weights = importances / importances.sum()

    # Wyświetlenie wag
    st.write("🔢 Proponowane wagi portfela:")
    weights_df = pd.DataFrame({
        "Spółka": selected_stocks,
        "Waga (%)": (weights * 100).round(2)
    }).sort_values("Waga (%)", ascending=False)
    st.dataframe(weights_df, use_container_width=True)

    # Wizualizacja portfela
    fig, ax = plt.subplots()
    ax.pie(weights, labels=selected_stocks, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

else:
    st.warning("Wybierz przynajmniej jedną spółkę.")
