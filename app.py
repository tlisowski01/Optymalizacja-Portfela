import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas_datareader.data as web



st.set_page_config(layout="wide")

st.title("📈 AI Optymalizacja Portfela - Spółki WIG20")

# Lista spółek WIG20 z tickerami na Yahoo Finance (odpowiedniki dla Stooq nie działają z yfinance)
wig20_tickers = {
    "PKN Orlen": "PKN",
    "PKO BP": "PKO",
    "PZU": "PZU",
    "KGHM": "KGH",
    "Santander": "SPL",
    "CD Projekt": "CDR",
    "JSW": "JSW",
    "PGE": "PGE",
    "Tauron": "TPE",
    "Cyfrowy Polsat": "CPS",
    "LPP": "LPP",
    "Alior Bank": "ALR",
    "Asseco Poland": "ACP",
    "mBank": "MBK",
    "Orange": "OPL",
    "Dino": "DNP",
    "CCC": "CCC",
    "Pepco": "PCO",
    "Grupa Kęty": "KTY",
    "Allegro": "ALE"
}


def get_stooq_data(ticker, start, end):
    try:
        df = web.DataReader(f"{ticker}.WA", 'stooq', start, end)
        df = df[::-1]  # Odwróć, bo Stooq zwraca dane od najnowszych do najstarszych
        return df["Close"]  # lub 'Open', 'Low', 'High', jak wolisz
    except Exception as e:
        st.error(f"❌ Błąd pobierania danych dla {ticker}: {e}")
        return None


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
        df = get_stooq_data(ticker, start_date, end_date)
        if df is not None:
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
