# Header container libraries
import streamlit as st
import pandas as pd
import numpy as np
import fredapi
from fredapi import Fred
import ssl
ssl._create_default_https_context = ssl._create_unverified_context 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Stock_Chart container libraries
from yahoo_fin import stock_info as si
# import yahoo_fin.stock_info as si
import plotly.graph_objs as go
import yfinance as yf 
# import pandas as pd
from datetime import date


pd.set_option('display.max_colwidth', -1)
pd.set_option('expand_frame_repr', False)

fred = Fred(api_key='25a7a6cc1d260aac37a4c0b6f7acac58')

st.set_page_config(
    page_title="QuantStacks Beta",
    page_icon="🇺🇸"
)

# CONTAINERS
header = st.container()
stock_chart = st.container()
dcf_valuation = st.container()


# HEADER CONTAINER
with header:
    st.title('QuantStacks Beta')
    # st.header('Buffett Indicator')

    # Pull Whilshire 5000 Price data
    wilshire_price = fred.get_series('WILL5000PR', observation_start='1970-01-01')
    wilshire_price = pd.DataFrame(wilshire_price, columns={'wilshire5K'})
    wilshire_price = wilshire_price.dropna()
    # st.write(wilshire_price.tail())

    # Pull USA GDP data and set frequency data to daily and ffill data
    gdp_data = fred.get_series('GDP', observation_start='1970-01-01').asfreq('d').ffill()
    gdp_data = pd.DataFrame(gdp_data, columns={'GDP'})
    # st.write(gdp_data.tail())

    # Concat DataFrame to calculate Buffett Indicator
    combined = pd.concat([wilshire_price, gdp_data], axis=1).dropna()
    combined['buffett_ind'] = combined['wilshire5K'] / combined['GDP']
    # st.write(combined.tail())

    # Calculate Buffett Indicator stats
    stats = combined['buffett_ind'].describe()
    # st.write(stats)

    # Plot Buffett Indicator
    combined['buffett_ind'].plot(figsize=(16,8), title='Buffett Indicator = Wilshire 5000 Price Index / USA GDP', grid=True, xlabel='Date', ylabel='%', c='b')
    plt.axhline(stats['mean'], c='y', label='Mean')
    plt.axhline(stats['50%'], label='Mode', c='c')
    plt.axhline(stats['25%'], label='25%', c='g')
    plt.axhline(stats['75%'], label='75%', c='r')
    plt.legend()

    with st.expander('Buffett Indicator'):
        st.pyplot(plt)

# STOCK_CHART CONTAINER
with stock_chart:
    # Get ticker list
    sp500_tickers = si.tickers_sp500()
    nasdaq_tickers = si.tickers_nasdaq()
    dow_tickers = si.tickers_dow()
    other_tickers = si.tickers_other()
    usa_tickers = sp500_tickers + nasdaq_tickers + dow_tickers + other_tickers
    usa_tickers.insert(0, 'Spy')
    usa_tickers.insert(1, 'All')

    # Dropdown box for ticker bar
    selected_option = st.multiselect(
        label='Enter ticker below',
        options=usa_tickers,
        default=["Spy"],
        )

    if "All" in selected_option:
        selected_option = usa_tickers[1:]

    for stock in selected_option:
        df = yf.download(stock, period='max')

        # Get yahoo minute data
        data = yf.download(tickers=stock, period='1d', interval='1m')

        # Graph stock chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name ='market data'
            ))

        fig.update_layout(
            title = stock,
            yaxis_title = 'Stock Price (per share)')

        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
            buttons = list([
                dict(count = 1, label='1m', step='minute', stepmode='backward'),
                dict(count = 5, label='5m', step='minute', stepmode='backward'),
                dict(count = 15, label='15m', step='minute', stepmode='backward'),
                dict(count = 30, label='30m', step='minute', stepmode='backward'),
                dict(count = 1, label='HTD', step='hour', stepmode='todate')
            ])))
        
        # st.plotly_chart(fig, use_container_width=True)
        with st.expander(stock + " Stock Chart", expanded=True ):
            st.plotly_chart(fig, expanded=False, use_container_width=True)

# DCF VALUATION
with dcf_valuation:
    #### DISCOUNT RATE (WACC) ####
    ### Cost of Equity ###

    ## Risk Free Rate ##

    # Scrape counties 10 years bond rate

    # 