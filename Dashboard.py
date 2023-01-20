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

# Fundamental_Data container libraries
from datetime import datetime
import datetime as dt
import pandas_datareader.data as reader
import lxml
from lxml import html
import requests
import numpy as np
# import pandas as pd
# import yfinance as yf
# import yahoo_fin.stock_info as si


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
fundamental_data = st.container()
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

# FUNDAMENTAL DATA
with fundamental_data:
    # Get Yahoo page
    def get_page(url):
    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'close',
            'DNT': '1', # Do Not Track Request Header 
            'Pragma': 'no-cache',
            'Referrer': 'https://google.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        }

        return requests.get(url, headers=headers)

    # yahoo parse rows
    def parse_rows(table_rows):
        parsed_rows = []

        for table_row in table_rows:
            parsed_row = []
            el = table_row.xpath("./div")

            none_count = 0

            for rs in el:
                try:
                    (text,) = rs.xpath('.//span/text()[1]')
                    parsed_row.append(text)
                except ValueError:
                    parsed_row.append(np.NaN)
                    none_count += 1

            if (none_count < 4):
                parsed_rows.append(parsed_row)
                
        return pd.DataFrame(parsed_rows)

    # yahoo clean data
    def clean_data(df):
        df = df.set_index(0) # Set the index to the first column: 'Period Ending'.
        df = df.transpose() # Transpose the DataFrame, so that our header contains the account names
        
        # Rename the "Breakdown" column to "Date"
        cols = list(df.columns)
        cols[0] = 'Date'
        df = df.set_axis(cols, axis='columns', inplace=False)
        
        numeric_columns = list(df.columns)[1::] # Take all columns, except the first (which is the 'Date' column)

        for column_index in range(1, len(df.columns)): # Take all columns, except the first (which is the 'Date' column)
            df.iloc[:,column_index] = df.iloc[:,column_index].str.replace(',', '') # Remove the thousands separator
            df.iloc[:,column_index] = df.iloc[:,column_index].astype(np.float64) # Convert the column to float64
            
        return df

    # yahoo scrape table
    def scrape_table(url):
        # Fetch the page that we're going to parse
        page = get_page(url);

        # Parse the page with LXML, so that we can start doing some XPATH queries
        # to extract the data that we want
        tree = html.fromstring(page.content)

        # Fetch all div elements which have class 'D(tbr)'
        table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
        
        # Ensure that some table rows are found; if none are found, then it's possible
        # that Yahoo Finance has changed their page layout, or have detected
        # that you're scraping the page.
        assert len(table_rows) > 0
        
        df = parse_rows(table_rows)
        df = clean_data(df)
            
        return df

    # Get ticker list
    sp500_tickers = si.tickers_sp500()
    nasdaq_tickers = si.tickers_nasdaq()
    dow_tickers = si.tickers_dow()
    other_tickers = si.tickers_other()
    usa_tickers = sp500_tickers + nasdaq_tickers + dow_tickers + other_tickers
    usa_tickers.insert(0, 'All')

    # Dropdown box for ticker bar
    selected_option = st.multiselect(
        label='Enter ticker below',
        options=usa_tickers,
        default=["TSLA"],
        )

    if "All" in selected_option:
        selected_option = usa_tickers[1:]

    # for stock in selected_option:
    def dcf_valuation(stock):
        try:
            try:
                # Get Risk Free Rate
                countries_10_years_bond_rate = pd.read_html("http://www.worldgovernmentbonds.com/", header=0)[1]

                if stock.endswith(('.SS', '.SZ')):
                    # China 10 years bond
                    filter_china_10y_bond = (countries_10_years_bond_rate['Unnamed: 1'] == 'China')
                    china_10years_bond = countries_10_years_bond_rate[filter_china_10y_bond]
                    china_10years_bond = china_10years_bond.iloc[0]['10Y Bond'] 
                    china_10years_bond = float(china_10years_bond.replace("%","")) /100
                    # st.write("china_10years_bond = {}".format(china_10years_bond))

                    # China Default Spread
                    filter_china_default_spread = (countries_10_years_bond_rate['Unnamed: 1'] == 'China')
                    china_default_spread = countries_10_years_bond_rate[filter_china_default_spread]
                    china_default_spread = china_default_spread.iloc[0]['Spread vs.1']
                    china_default_spread = float(china_default_spread.replace("bp","")) /10000
                    # print("china_default_spread = {}".format(china_default_spread)) 

                    # China Risk Free Rate
                    risk_free_rate = china_10years_bond - china_default_spread
                    # st.write("risk_free_rate = {}".format(risk_free_rate))
                    # st.write("China's Risk Free Rate")
                    # st.write(risk_free_rate)

                else:
                    # USA 10 years bond
                    filter_usa_10years_bond = (countries_10_years_bond_rate['Unnamed: 1'] == 'United States')
                    usa_10years_bond = countries_10_years_bond_rate[filter_usa_10years_bond]
                    usa_10years_bond = usa_10years_bond.iloc[0]['10Y Bond'] 
                    usa_10years_bond = float(usa_10years_bond.replace("%","")) /100
                    # print("usa_10years_bond = {}".format(usa_10years_bond))

                    # USA Risk Free Rate
                    risk_free_rate = usa_10years_bond
                    # print("risk_free_rate = {}".format(risk_free_rate))
                    # st.write("USA's Risk Free Rate")
                    # st.write(risk_free_rate)
            except:
                st.write('risk free rate failed')
                pass
            # get balance sheet
            try:
                df_balance_sheet = scrape_table('https://finance.yahoo.com/quote/' + stock + '/balance-sheet?p=' + stock)
                df_balance_sheet = df_balance_sheet.set_index('Date')
                # st.write(stock + ' Balance Sheet')
                # df_balance_sheet
            except:
                st.write(stock + ' balance sheet failed')
                pass 

            # get income statement
            try:
                df_income_statement = scrape_table('https://finance.yahoo.com/quote/' + stock + '/financials?p=' + stock)
                df_income_statement = df_income_statement.set_index('Date')
                # st.write(stock + ' Income Statement')
                # df_income_statement
            except:
                st.write(stock + ' income statement failed')
                pass 

            # get cash flow statement
            try:
                df_cash_flow = scrape_table('https://finance.yahoo.com/quote/' + stock + '/cash-flow?p=' + stock)
                df_cash_flow = df_cash_flow.set_index('Date')
                # st.write(stock + ' Cash Flow Statement')
                # df_cash_flow
            except:
                st.write(stock + ' cash flow statement failed')
                pass

            # Get quote table
            try:
                quote = si.get_quote_table(stock) 
                # st.write(quote)
            except:
                st.write('quote table failed')
                pass

            # Get Beta
            try:
                # st.text('Beta')
                beta = float(quote['Beta (5Y Monthly)'])
                # st.write(beta)
            except:
                pass

            # get Market Cap
            try:
                # st.write('Market Cap')
                mc = str(quote['Market Cap'])
                if mc[-1] == 'T':
                    fmc = float(mc.replace('T',''))
                    marketCap = fmc*1000000000000
                    # st.write(marketCap)
                elif mc[-1] == 'B':
                    fmc = float(mc.replace('B',''))
                    marketCap = fmc*1000000000
                    # st.write(marketCap)
                elif mc[-1] == 'M':
                    fmc = float(mc.replace('M',''))
                    marketCap = fmc*1000000
                    # st.write(marketCap) 
            except:
                pass

            # Get Total Debt
            try:
                Total_Debt = df_balance_sheet['Total Debt'][0]
                # st.write(stock + ' Total Debt')
                # st.write(Total_Debt)
            except:
                st.write(stock + ' Total Debt Failed!')
                pass

            # Calculate Weight of Equity
            try:
                Weight_of_Equity = marketCap / (marketCap + Total_Debt)
                # st.write(stock + ' Weight of Equity')
                # st.write(Weight_of_Equity)
            except:
                st.write(stock + ' Weight of Equity Failed!')
                pass 

            # Calculate Weight of Debt
            try:
                Weight_of_Debt = Total_Debt / (marketCap + Total_Debt)
                # st.write(stock + ' Weight of Debt')
                # st.write(Weight_of_Debt)                
            except:
                st.write(stock + ' Weight of Debt Failed!')
                pass

            # Get Interest Expense  
            try:
                Interest_Expenses = df_income_statement['Interest Expense'][0]
                # st.write(stock + ' Interest Expense')
                # st.write(Interest_Expenses)
            except:
                st.write(stock + ' Interest Expense Failed!')
                pass

            # Get Income Tax Expense
            try:
                Income_Tax_Expense = df_income_statement['Tax Provision'][0]
                # st.write(stock + ' Income_Tax_Expense')
                # st.write(Income_Tax_Expense)                
            except:
                st.write(stock + ' Income Tax Expense Failed!')
                pass 

            # Get Income Before Tax
            try:
                Income_Before_Tax = df_income_statement['Pretax Income'][0]
                # st.write(stock + ' Income_Before_Tax')
                # st.write(Income_Before_Tax)                
            except:
                st.write(stock + ' Income_Before_Tax Failed')
                pass 

            # Calculate effective tax rate
            try:
                Effective_Tax_Rate = Income_Tax_Expense / Income_Before_Tax
                # st.write(stock + ' Effective Tax Rate')
                # st.write(Effective_Tax_Rate)
            except:
                st.write(stock + ' Effective Tax Rate Failed')
                pass

            # Calculate: Cost of Debt = Interest Expenses / Total Debt
            try:
                Cost_of_Debt = Interest_Expenses / Total_Debt
                # st.write(stock + ' Cost of Debt')
                # st.write(Cost_of_Debt)
            except:
                st.write(stock + ' Cost of Debt Failed')
                pass

            # Cost of Debt(1-t) = Cost of Debt * (1 - Effective Tax Rate)
            try:
                Cost_of_Debt_1t = Cost_of_Debt * (1 - Effective_Tax_Rate)
                # st.write(stock + ' Cost of Debt(1-t)')
                # st.write(Cost_of_Debt_1t)
            except:
                st.write(stock + ' Cost of Debt(1-t) Failed')
                pass 
            
            # Interest Coverage Ratio (Estimating Synthetic Ratings) = EBIT / Interest Expenses
            try:
                Interest_Coverage_Ratio = df_income_statement['EBIT'][0] / Interest_Expenses
                # st.write(stock + ' Interest Coverage Ratio')
                # st.write(Interest_Coverage_Ratio)
            except:
                st.write(stock + ' Interest Coverage Ratio Failed')
                pass

            # Calculate: Market Return = [(Ending Price - Beginning Price) / (Beginning Price)] + [(Dividend) / (Begining Price)] 
            
            # Calculate: Cost of Equity = Risk Free Rate + Beta(Market Return - Risk Free Rate)

            # Discount Rate (WACC)
            try:
                Discount_Rate_WACC = (Weight_of_Equity * Cost_of_Equity) + (Weight_of_Debt * Cost_of_Debt) * (1 - Effective_Tax_Rate)
            except:
                pass 

            # Analyst Growth Estimate
            try:
                stock = stock.upper()
                analysts = si.get_analysts_info(stock)
                analysts_growth_estimate = analysts['Growth Estimates'][stock][5]
                analysts_growth_estimate = float(analysts_growth_estimate.replace("%","")) /100
                # st.write(stock + ' Analysts Growth Estimate')
                # st.write(analysts_growth_estimate)
            except:
                st.write(stock + ' Analysts Growth Estimate Failed')
                pass

            # Symbol current price
            try:
                current_price = si.get_live_price(stock)
                # st.write(stock + ' Current Price')
                # st.write(current_price)
            except:
                st.write(stock + ' current price failed')
                pass

            # ttm cash flow
            try:
                ttm_cashflow = current_price * risk_free_rate
                # st.write(stock + ' ttm cashflow')
                # st.write(ttm_cashflow)
            except:
                st.write(stock + ' ttm cashflow failed')
                pass

            # projected cashflow
            try:
                years = [1,2,3,4,5]
                futurefreecashflow = []
                for year in years:
                    cashflow = ttm_cashflow * (1 + analysts_growth_estimate)**year
                    futurefreecashflow.append(cashflow)
                # st.write(stock + ' projected cashflow statement')
                # st.write(futurefreecashflow)
            except:
                st.write(stock + ' projected cashflow failed')
                pass

            # Expected Return on symbol
            try:
                from scipy import optimize

                def fun(r):
                    r1 = 1 + r
                    return futurefreecashflow[0]/r1 +  futurefreecashflow[1]/r1**2 + futurefreecashflow[2]/r1**3 + futurefreecashflow[3]/r1**4 + futurefreecashflow[4]/r1**5 * (1 + (1+risk_free_rate)/(r-risk_free_rate)) - current_price

                roots = optimize.root(fun, [.1])
                expected_return_on_stock = float(roots.x)
                # st.write(stock + ' Expected Return on Stock')
                # st.write(expected_return_on_stock)
            except:
                st.write(stock + ' Expected Return on Stock failed')
                pass
            
            # Implied Equity Risk Premium
            try:
                implied_equity_risk_premium = expected_return_on_stock - risk_free_rate
                # st.write(stock + ' Implied Equity Risk Premium')
                # st.write(implied_equity_risk_premium)
            except:
                st.write(stock + ' implied equity rick premium failed')
                pass 

            dcf_list = [implied_equity_risk_premium]
            return dcf_list

        except:
            st.write('dcf_valuation failed')
            pass
    
    dcf_table = []
    temp_table = []

    for stock in selected_option:
        temp_table = dcf_valuation(stock)
        temp_table.insert(0, stock)
        dcf_table.append(temp_table)

    dcf_column_name=['Symbol', 'Implied Equity Risk Piremium']
    dcf_index = range(len(selected_option))
    dcf_df = pd.DataFrame(data=dcf_table, index=dcf_index,columns=dcf_column_name)
    with st.expander(" DCF Valuation", expanded=True):
        st.table(dcf_df)