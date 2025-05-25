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

# Updated list of WIG20 companies with correct Yahoo Finance tickers
wig20_tickers = {
    "PKN Orlen": "PKN.WA",
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

# User selects stocks
selected_stocks = st.multiselect("Wybierz spółki z WIG20:", list(wig20_tickers.keys()))

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)

# Download data function
@st.cache_data
def download_data(tickers):
    data = {}
    for name in tickers:
        ticker = wig20_tickers[name]
        try:
            # Pobieramy pełne dane
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if stock_data.empty:
                st.warning(f"Brak danych dla {name} ({ticker})")
                continue
                
            # Zapisujemy tylko dane o zamknięciu
            data[name] = stock_data['Adj Close']
            
        except Exception as e:
            st.error(f"Błąd podczas pobierania {name} ({ticker}): {str(e)}")
            continue
    
    if data:
        return pd.DataFrame(data)
    return None

if selected_stocks:
    st.info("Pobieranie danych...")
    prices = download_data(selected_stocks)
    
    if prices is not None and not prices.empty:
        st.success("Dane pobrane pomyślnie!")
        
        # Display price chart
        st.subheader("📊 Wykres cen")
        st.line_chart(prices)
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        st.subheader("🔍 Optymalizacja portfela (AI)")
        
        # Prepare data for AI
        X = returns.copy()
        y = returns.mean(axis=1)  # artificial metric for regression
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        weights = importances / importances.sum()
        
        # Display weights
        st.write("🔢 Proponowane wagi portfela:")
        weights_df = pd.DataFrame({
            "Spółka": selected_stocks,
            "Waga (%)": (weights * 100).round(2)
        }).sort_values("Waga (%)", ascending=False)
        st.dataframe(weights_df, use_container_width=True)
        
        # Portfolio visualization
        fig, ax = plt.subplots()
        ax.pie(weights, labels=selected_stocks, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.error("Nie udało się pobrać danych dla wybranych spółek. Spróbuj ponownie.")
else:
    st.warning("Wybierz przynajmniej jedną spółkę.")
