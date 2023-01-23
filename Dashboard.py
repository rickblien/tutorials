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

st.title('QUANTSTACKS')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Overview", "News", "DCF Valuation", "P/E Ratio Bell Curve", "VaR"])

# Market Overview
with tab1:
    # Buffett Indicator
    st.header("Buffett Indicator")

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

    with st.expander('Buffett Indicator', expanded=True):
        st.pyplot(plt)

    # S&P 500 Chart
    st.header('S&P 500 Index')
    # S&P 500 Chart Libraries
    import yfinance as yf
    import plotly.graph_objs as go

    # S&P 500 ticker
    stock = '^GSPC'
    data = yf.download(tickers=stock, period='1d', interval='1m')
    data.tail()

    # S&P 500 Chart
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
        title = 'Live Market Data',
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
# News
with tab2:
   st.header("News")
   st.write('Coming soon!')

# DCF VALUATION
with tab3:
    st.header("DCF Valuation")

    # SEARCH BAR
    # Get tickers list
    from yahoo_fin import stock_info as si
    import pandas as pd

    # Get USA ticker list
    df1 = pd.DataFrame(si.tickers_sp500())
    df2 = pd.DataFrame(si.tickers_nasdaq())
    df3 = pd.DataFrame(si.tickers_dow())
    df4 = pd.DataFrame(si.tickers_other())

    # convert DataFrame to list, then to sets
    sym1 = set(symbol for symbol in df1[0].tolist())
    sym2 = set(symbol for symbol in df2[0].tolist())
    sym3 = set(symbol for symbol in df3[0].tolist())
    sym4 = set(symbol for symbol in df4[0].tolist())

    # join the 4 sets into one. Because it's a set, there will be no duplicate symbols
    symbols = set.union(sym1, sym2, sym3, sym4)

    # some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest
    my_list = ['W', 'R', 'P', 'Q']
    del_set = set()
    sav_set = set()

    for symbol in symbols:
        if len( symbol ) > 4 and symbol[-1] in my_list:
            del_set.add( symbol )
        else:
            sav_set.add( symbol )
    
    sav_set = sorted(sav_set)
    sav_set.insert(0, 'ALL')
    usa_tickers = sav_set

    # Dropdown box for ticker bar
    selected_option = st.multiselect(
        label='Enter ticker below',
        options=usa_tickers,
        default=["TSLA"],
        )

    if "All" in selected_option:
        selected_option = usa_tickers[1:]

    # DCF VALUATION
    def dcf_valuation(stock):
        try:
            # Calculate: risk_free_rate
            def risk_free_rate(country_option):
                try:
                    # Filter out selected country 10 years bond
                    filter_country_10y_bond = (countries_10_years_bond_rate['country_name'] == country_option)
                    country_10years_bond = countries_10_years_bond_rate[filter_country_10y_bond]
                    country_10years_bond = country_10years_bond.iloc[0]['10y_bond'] 
                    country_10years_bond = float(country_10years_bond.replace("%","")) /100
                    # st.write(country_10years_bond)

                    # Calculate Default Spread
                    # st.header('Default Spread')
                    # st.write("Country Default Spread = Country 10 Years Bond - USA 10 Years Bond")

                    # USA 10 years bond
                    filter_usa_10years_bond = (countries_10_years_bond_rate['country_name'] == 'United States')
                    usa_10years_bond = countries_10_years_bond_rate[filter_usa_10years_bond]
                    usa_10years_bond = usa_10years_bond.iloc[0]['10y_bond'] 
                    usa_10years_bond = float(usa_10years_bond.replace("%","")) /100
                    # st.write('USA 10 years bond is below')
                    # st.write(usa_10years_bond)

                    # Selected Country Default Spread
                    country_default_spread_usd = country_10years_bond - usa_10years_bond
                    # st.write('Selected Country Default Spread is below')
                    # st.write(country_default_spread_usd)
                    # st.write("Country Default Spread = Country 10 Years Bond - USA 10 Years Bond")

                    # Calculate Selected Country Risk Free Rate
                    country_risk_free_rate = usa_10years_bond - country_default_spread_usd
                    # st.write("Selected country's risk free rate")
                    # st.write(country_risk_free_rate)
                    # st.write("Country Risk Free Rate = USA 10 Years Bond - Country Default Spread")
                    
                    listd = [country_10years_bond, usa_10years_bond,country_default_spread_usd,country_risk_free_rate]
                    return listd
                    
                except:
                    # st.write(country_option + "failed!")
                    pass

            # Scrape Countries 10 years government bond rates
            import pandas as pd
            import numpy as np
            countries_10_years_bond_rate = pd.read_html("http://www.worldgovernmentbonds.com/", header=0)[1]
            countries_10_years_bond_rate.rename(columns={"Unnamed: 0":"Nan","Unnamed: 1":"country_name","Rating":"rating","10Y Bond":"10y_bond","10Y Bond.1":"10y_bond_1","Bank":"bank", "Spread vs":"spread_vs", "Spread vs.1":"spread_vs_1"},inplace=True)
            # country_name = countries_10_years_bond_rate['country_name'][1:].tolist() 

            country_data_in_table = []
            temp_table = []

            if stock.endswith(('.SS', '.SZ')):
                country_option = ['China']
            else:
                country_option = ['United States']

            for country in country_option:
                temp_table = risk_free_rate(country)
                temp_table.insert(0,country)
                country_data_in_table.append(temp_table)
            
            colu = ["Country", "10years bond", "USA 10years bond","Country Default Spread", "Risk Free Rate"]
            inde = range(len(country_option))
            risk_free_rate_df = pd.DataFrame(data= country_data_in_table,index=inde,columns=colu)
            # risk_free_rate_df

            # risk_free_rate
            risk_free_rate = risk_free_rate_df['Risk Free Rate'][0]
            # st.write(stock + ' risk_free_rate')
            # st.write(risk_free_rate)

            # country_default_spread
            country_default_spread = risk_free_rate_df['Country Default Spread'][0]
            # st.write(stock + ' country_default_spread')
            # st.write(country_default_spread)

            # Get quote table
            try:
                quote = si.get_quote_table(stock) 
                # st.write(stock + ' quote table')
                # st.write(quote)
            except:
                # st.write('quote table failed')
                pass

            # Get Beta
            try:
                # st.text('Beta')
                beta = float(quote['Beta (5Y Monthly)'])
                # st.write(stock + ' beta')
                # st.write(beta)
            except:
                pass

            # Get: current_price
            try:
                current_price = si.get_live_price(stock)
                # st.write(stock + ' Current Price')
                # st.write(current_price)
            except:
                # st.write(stock + ' current price failed')
                pass

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
                
                # numeric_columns = list(df.columns)[1::] # Take all columns, except the first (which is the 'Date' column)

                for column_index in range(1, len(df.columns)): # Take all columns, except the first (which is the 'Date' column)
                    df.iloc[:,column_index] = df.iloc[:,column_index].str.replace(',', '') # Remove the thousands separator
                    df.iloc[:,column_index] = df.iloc[:,column_index].astype(np.float64) # Convert the column to float64
                    
                return df

            # yahoo scrape table
            def scrape_table(url):
                # Fetch the page that we're going to parse
                page = get_page(url)

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

            # get balance sheet
            try:
                df_balance_sheet = scrape_table('https://finance.yahoo.com/quote/' + stock + '/balance-sheet?p=' + stock)
                df_balance_sheet = df_balance_sheet.set_index('Date')
                # st.write(stock + ' df_balance_sheet')
                # df_balance_sheet
            except:
                # st.write(stock + ' df_balance_sheet failed')
                pass 

            # get income statement
            try:
                df_income_statement = scrape_table('https://finance.yahoo.com/quote/' + stock + '/financials?p=' + stock)
                df_income_statement = df_income_statement.set_index('Date')
                # st.write(stock + ' df_income_statement')
                # df_income_statement
            except:
                # st.write(stock + ' df_income_statement failed')
                pass 

            # get cash flow statement
            try:
                df_cash_flow = scrape_table('https://finance.yahoo.com/quote/' + stock + '/cash-flow?p=' + stock)
                df_cash_flow = df_cash_flow.set_index('Date')
                # st.write(stock + ' df_cash_flow')
                # df_cash_flow
            except:
                # st.write(stock + ' df_cash_flow failed')
                pass

            # Get: ttm_cashflow
            # ttm_cashflow = free cash flow / number of share outstanding
            # 1) ttm_cashflow = operating cash flow - capital expenditure
            # 2) ttm_cashflow = sales revenue - (operating costs + taxes) - required investments in operating capital
            # 3) ttm_cashflow = net operating profit after taxes - net investment in operating capital

            try:
                free_cash_flow = float(df_cash_flow['Free Cash Flow'][0] * 1000)
                ordinary_shares_number = float(df_balance_sheet['Ordinary Shares Number'][0] * 1000)
                ttm_cashflow = free_cash_flow / ordinary_shares_number
                # st.write(stock + ' ttm_cashflow')
                # st.write(ttm_cashflow)
            except:
                # st.write(stock + ' ttm_cashflow failed')
                pass

            # Get: analyst_5yrs_estimate_growth_rate
            try:
                stock = stock.upper()
                analysts = si.get_analysts_info(stock)
                analysts_growth_estimate = analysts['Growth Estimates'][stock][4]
                analysts_growth_estimate = float(analysts_growth_estimate.replace("%","")) /100
                # st.write(stock + ' analysts_growth_estimate')
                # st.write(analysts_growth_estimate)
            except:
                # st.write(stock + ' analysts_growth_estimate Failed')
                pass

            # Calculate: futurefreecashflow
            try:
                years = [1,2,3,4,5]
                futurefreecashflow = []
                for year in years:
                    cashflow = ttm_cashflow * (1 + analysts_growth_estimate)**year
                    futurefreecashflow.append(cashflow)
                # st.write(stock + ' futurefreecashflow')
                # st.write(futurefreecashflow)
            except:
                # st.write(stock + ' futurefreecashflow failed')
                pass

            # Calculate: expected_returns
            try:
                from scipy import optimize

                def fun(r):
                    r1 = 1 + r
                    return futurefreecashflow[0]/r1 +  futurefreecashflow[1]/r1**2 + futurefreecashflow[2]/r1**3 + futurefreecashflow[3]/r1**4 + futurefreecashflow[4]/r1**5 * (1 + (1+risk_free_rate)/(r-risk_free_rate)) - current_price

                roots = optimize.root(fun, [.1])
                expected_returns = float(roots.x)
                # st.write(stock + ' expected_returns')
                # st.write(expected_returns)
            except:
                # st.write(stock + ' expected_returns failed')
                pass

            # Calculate: implied_equity_risk_premium
            try:
                implied_equity_risk_premium = expected_returns - risk_free_rate
                # st.write(stock + ' implied_equity_risk_premium')
                # st.write(implied_equity_risk_premium)
            except:
                # st.write(stock + ' implied_equity_risk_premium failed')
                pass 

            # Calculate: cost_of_equity (CAPM) = Risk Free Rate + beta * (Market Premium - Risk Free Rate)
            try:
                cost_of_equity = risk_free_rate + beta * (implied_equity_risk_premium)
                # st.write(stock + ' cost_of_equity')
                # st.write(cost_of_equity)
            except:
                # st.write(stock + ' cost_of_equity failed')
                pass

            # Get: market_cap
            try:
                # st.write('market_cap')
                mc = str(quote['Market Cap'])
                if mc[-1] == 'T':
                    fmc = float(mc.replace('T',''))
                    market_cap = fmc*1000000000000
                    # st.write(market_cap)
                elif mc[-1] == 'B':
                    fmc = float(mc.replace('B',''))
                    market_cap = fmc*1000000000
                    # st.write(market_cap)
                elif mc[-1] == 'M':
                    fmc = float(mc.replace('M',''))
                    market_cap = fmc*1000000
                    # st.write(market_cap) 
            except:
                # st.write(stock + ' market_cap failed')
                pass

            # Get: total_debts
            try:
                total_debt = df_balance_sheet['Total Debt'][0] * 1000
                # st.write(stock + ' total_debt')
                # st.write(total_debt)
            except:
                # st.write(stock + ' total_debt Failed!')
                pass

            # Calculate: total_equity
            try:
                total_equity = market_cap - total_debt
                # st.write(stock + ' total_equity')
                # st.write(total_equity)
            except:
                # st.write(stock + ' total_equity failed')
                pass 

            # Calculate: weighted_equity
            try:
                weight_equity = market_cap / (market_cap + total_debt)
                # st.write(stock + ' weight_equity')
                # st.write(weight_equity)
            except:
                # st.write(stock + ' weight_equity Failed!')
                pass 

            # Get: interest_expenses
            try:
                interest_expenses = df_income_statement['Interest Expense'][0] * 1000
                # st.write(stock + ' interest_expenses')
                # st.write(interest_expenses)
            except:
                # st.write(stock + ' interest_expenses Failed!')
                pass

            # Get: ebit
            try:
                ebit = df_income_statement['EBIT'][0] * 1000
                # st.write(stock + ' ebit')
                # st.write(ebit)
            except:
                # st.write(stock + ' ebit failed')
                pass

            # Calculate: interest_coverage_ratio (Estimating Synthetic Ratings) = ebit / interest_expense
            try:
                interest_coverage_ratio = ebit / interest_expenses
                # st.write(stock + ' interest_coverage_ratio')
                # st.write(interest_coverage_ratio)
            except:
                # st.write(stock + ' interest_coverage_ratio Failed')
                pass

            # Get: company_default_spread_table for interest_coverage_ratio
            try:
                import pandas as pd
                import ssl
                import numpy as np
                ssl._create_default_https_context = ssl._create_unverified_context
                ratings = pd.read_html("https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html", header=0)
                ratings_data = np.array(ratings)

                ratings_table = []

                for rating in ratings_data:
                    for rating_data in rating:
                        ratings_table.append(rating_data)

                colu = ["Item1", "Item2", "Item3","Item4"]

                inde = range(len(ratings_table))

                rat = pd.DataFrame(data= ratings_table,index=inde,columns=colu)

                # st.write(rat)
                score = interest_coverage_ratio
                match = (rat['Item1'] <= score) & (rat['Item2'] > score)
                company_default_spread = float(rat['Item4'][match].values[0].replace("%","")) / 100
                # st.write(stock + ' company_default_spread')
                # st.write(company_default_spread)
            except:
                # st.write(stock + ' company_default_spread failed')
                pass

            # Calculate: cost_of_debt = risk_free_rate + [2/3 * (country_default_spread)] + company_default_spread
            try:
                cost_of_debt = (risk_free_rate + [2/3 * (country_default_spread)] + company_default_spread)[0]
                # st.write(stock + ' cost_of_debt')
                # st.write(cost_of_debt)
            except:
                # st.write(stock + ' cost_of_debt Failed')
                pass 

            # Get: income_tax_expense
            try:
                income_tax_expense = df_income_statement['Tax Provision'][0] * 1000
                # st.write(stock + ' income_tax_expense')
                # st.write(income_tax_expense)                
            except:
                # st.write(stock + ' income_tax_expense Failed!')
                pass 

            # Get: income_before_tax
            try:
                income_before_tax = df_income_statement['Pretax Income'][0] * 1000
                # st.write(stock + ' income_before_tax')
                # st.write(income_before_tax)                
            except:
                # st.write(stock + ' income_before_tax Failed')
                pass 

            # Calculate: effective_tax_rate
            try:
                effective_tax_rate = income_tax_expense / income_before_tax
                # st.write(stock + ' effective_tax_rate')
                # st.write(effective_tax_rate)
            except:
                # st.write(stock + ' effective_tax_rate Failed')
                pass

            # Calculate: cost_of_debt = (risk_free_rate + ((2/3) * country_default_spread) + company_default_spread)
            try:
                cost_of_debt = (risk_free_rate + ((2/3) * country_default_spread) + company_default_spread)
                # st.write(stock + ' cost_of_debt')
                # st.write(cost_of_debt)
            except:
                # st.write(stock + ' cost_of_debt Failed')
                pass

            # Calculate: cost_of_debt_1-t
            try:
                cost_of_debt_1t = cost_of_debt * (1 - effective_tax_rate)
                # st.write(stock + ' cost_of_debt_1t')
                # st.write(cost_of_debt_1t)
            except:
                # st.write(stock + ' cost_of_debt_1t Failed')
                pass 

            # Calculate: weighted_debt
            try:
                weight_debt = total_debt / (market_cap + total_debt)
                # st.write(stock + ' weight_debt')
                # st.write(weight_debt)                
            except:
                # st.write(stock + ' weight_debt Failed!')
                pass

            # Calculate: discount_rate_wacc = (weight_debt * cost_of_debt_1t) + weight_equity * cost_of_equity)
            try:
                discount_rate_wacc = (weight_debt * cost_of_debt_1t) + (weight_equity * cost_of_equity)
                # st.write(stock + ' discount_rate_wacc')
                # st.write(discount_rate_wacc)
            except:
                # st.write(stock + ' discount_rate_wacc failed')
                pass

            # Calculate: terminal_value
            try:
                discountfactor = []

                terminal_value = (ttm_cashflow * ordinary_shares_number) * (1+risk_free_rate) / (discount_rate_wacc - risk_free_rate)
                # st.write(stock + ' terminal_value ')
                # st.write(terminal_value)  
            except:
                # st.write(stock + ' terminal_value failed')
                pass

            # Calculate: futurefreecashflow
            try:
                years = [1,2,3,4,5]
                futurefreecashflow = []
                for year in years:
                    cashflow = (ttm_cashflow * ordinary_shares_number) * (1+analysts_growth_estimate)**year
                    futurefreecashflow.append(cashflow)
                    discountfactor.append((1 + discount_rate_wacc)**year)
                # st.write(stock + ' futurefreecashflow')
                # st.write(futurefreecashflow)
            except:
                # st.write(stock + ' futurefreecashflow failed')
                pass

            # Calculate: discount_future_free_cashflow
            try:
                discountedfuturefreecashflow = []

                for i in range(0, len(years)):
                    discountedfuturefreecashflow.append(futurefreecashflow[i]/discountfactor[i])
                # st.write(stock + ' discountedfuturefreecashflow')
                # st.write(discountedfuturefreecashflow)
            except:
                # st.write(stock + ' discount_future_free_cashflow failed')
                pass
            
            # Calculate present value for terminal_value and added to discountedfuturefreecashflow
            try:
                discountedterminalvalue = terminal_value/(1 + discount_rate_wacc)**4
                discountedfuturefreecashflow.append(discountedterminalvalue)

            except:
                pass

            # Calculate: today_value
            try:
                today_value = sum(discountedfuturefreecashflow)
                # st.write(stock + ' today_value')
                # st.write(today_value)
            except:
                # st.write(stock + ' today_value failed')
                pass

            # Calculate fair_value
            try:
                fair_value = today_value / ordinary_shares_number
                # st.write(stock + ' fair_value')
                # st.write(fair_value)
            except:
                # st.write(stock + ' fair_value failed')
                pass


            dcf_list = [current_price, fair_value]
            return dcf_list

        except:
            pass    

    dcf_table = []
    dcf_temp_table = []

    for stock in selected_option:
        dcf_temp_table = dcf_valuation(stock)
        dcf_temp_table.insert(0, stock)
        dcf_table.append(dcf_temp_table)

    dcf_column_name=['Symbol', 'Current Price', 'DCF Value']
    dcf_index = range(len(selected_option))
    dcf_df = pd.DataFrame(data=dcf_table, index=dcf_index,columns=dcf_column_name)
    with st.expander(" DCF Valuation", expanded=True):
        st.table(dcf_df)

# P/E Ratio Bell Curve
with tab4:
    import streamlit as st
    from datetime import date
    import datetime as dt
    import yfinance as fyf
    import yahoo_fin.stock_info as si
    from pandas_datareader import data as pdr
    import talib
    import pandas as pd
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.stats import norm
    import math
    import plotly.figure_factory as ff
    import copy
    import ssl
    import plotly.graph_objects as go
    ssl._create_default_https_context = ssl._create_unverified_context

    st.title("PE Bell Curve")

    # start of the FrontEnd
    def main():
        """Runs the main user interface """
        # header = st.container()
        features = st.container()


        # with header:
        #     st.title('QuantStacks Prototype')
        #     st.text('Evaluate undervalued or overvalued stock.')

        with features:
            global ticker, start, end
            ticker =  st.text_input('Enter Desired Ticker')
            start = st.date_input("Enter Starting Date", datetime.date(1950,1,1)) # create start date input box, set '2020-02-11' = default
            today = date.today().strftime("%Y-%m-%d")
            end = st.text_input("End Date", f'{today}') # create end date input box, set today = default
            

            ticker = ticker.upper()
            if not ticker:
                st.info('Please input a valid ticker')
            else:
                st.success('You selected ticker: {}'.format(ticker))
                with st.spinner('Loading data......'):
                    setup()
                    OHLC_original = grab_OHLC(ticker)
                    Earnings = grab_historical_EPS(ticker)
                    #deepcopy OHLC to access full volume and volatility
                    OHLC = copy.deepcopy(OHLC_original)

                    edit_EPS(ticker,Earnings)
                    Merged = merge_OHLC_EPS_ttm(ticker, OHLC, Earnings)
                    #get PE only for the dates that have earnings
                    calc_PE_ratio(ticker,Merged)
                    #get Volatility of the full historical data
                    calc_volatility(ticker,OHLC_original)

                    st.header('PE ratio probability curve')
                    PE_stats = get_PE_stats(ticker,Merged)
                    PE_CDF = get_CDF(PE_stats)
                    st.subheader("Risk for {} is {}".format(ticker,PE_CDF))
                    suggestion(PE_CDF,"PE")
                    normal_distribution_curve(PE_stats,"PE Ratio")

                    st.header('Buy and Sell Signals')
                    morning_star = talib.CDLMORNINGSTAR(Historical_Prices['Open'], Historical_Prices['High'], Historical_Prices['Low'], Historical_Prices['Close'])
                    engulfing = talib.CDLENGULFING(Historical_Prices['Open'], Historical_Prices['High'], Historical_Prices['Low'], Historical_Prices['Close'])
                    Historical_Prices['Morning Star'] = morning_star
                    Historical_Prices['Engulfing'] = engulfing
                    engulfing_buy_days = Historical_Prices[Historical_Prices['Engulfing'] ==100]
                    engulfing_sell_days = Historical_Prices[Historical_Prices['Engulfing'] ==-100]
                    st.write('Buy Days')
                    st.write(engulfing_buy_days.tail(3))
                    st.write('Sell Days')
                    st.write(engulfing_sell_days.tail(3))
                    
                    st.header('Volatility curve')
                    volatility_stats = get_volatility_stats(ticker,OHLC_original)
                    volatility_CDF = get_CDF(volatility_stats)
                    st.subheader("The CDF of the Volatility for {} is {}".format(ticker,volatility_CDF))
                    suggestion(volatility_CDF,"Volatility")
                    normal_distribution_curve(volatility_stats,"Volatility")

                    st.header('Volume curve')
                    volume_stats = get_volume_stats(ticker,OHLC_original)
                    volume_CDF = get_CDF(volume_stats)
                    st.subheader("The CDF of the Volume for {} is {}".format(ticker,volume_CDF))
                    suggestion(volume_CDF,"Volume")
                    normal_distribution_curve(volume_stats,"Volume")

                st.success('Done!')

    def setup():
        """ Create necessary global variables, gets current directory and create path where all files will be downloaded to """
        # local file locations
        cwd = os.getcwd()
        global path
        path = cwd + "/"
        global Fundamental_URL
        Fundamental_URL = ("https://www.macrotrends.net/stocks/charts/")

    def grab_OHLC(ticker):
        """Grabs OHLC data from Yahoo Finance and return it as a dataframe """

        fyf.pdr_override()
        global Historical_Prices
        Historical_Prices = pdr.get_data_yahoo(ticker, start, end, inplace=False)
        #st.write(Historical_Prices)
        return Historical_Prices

    def grab_historical_EPS(ticker):
        """Scrapes earnings from ycharts and return it as a dataframe' """
        df = pd.read_html(Fundamental_URL+ticker+"/"+ticker+"/eps-earnings-per-share-diluted")[1]
        all_earnings = pd.DataFrame(data = df, columns=None)
        all_earnings.columns = ['Date','EPS']
        all_earnings['Date'] = pd.to_datetime(all_earnings['Date'])
        all_earnings['EPS'] = all_earnings['EPS'].str.replace('$', '', regex=True).astype(float) # omits '$', please update if not working in future
        all_earnings.set_index('Date', inplace = True) # make date column the index
        all_earnings = all_earnings.sort_index() #make sure data is stored oldest at top, newest on bottom
        
        return all_earnings

    def edit_EPS(ticker,Earnings_df):
        """Read in Earning_df and create new column EPS_ttm and return updated Earning_df """
        Earnings_df['EPS_ttm'] = Earnings_df['EPS'].rolling(window=4,center=False).sum()
        return Earnings_df

    def merge_OHLC_EPS_ttm(ticker,OHLC_df,Earning_df):
        """Merge Historical_Prices and Earnings_df and return Merged_df """
        # frames = [OHLC_df,Earning_df] #list of the dataframes that are loaded

        #loop that assigns eps to proper trade dates
        for eps_day in Earning_df.index:
            # print(eps_day)
            for trade_day in OHLC_df.index:
                #print(trade_day)
                if eps_day <= trade_day:
                    OHLC_df.at[trade_day,'EPS_ttm'] = Earning_df.loc[eps_day]['EPS_ttm']
                    #print(ohlc_data.at[trade_day,'EPS'])
        OHLC_df.dropna(inplace=True)
        return OHLC_df

    def calc_PE_ratio(ticker,Merged_df):
        """Read in Merged_df and create a new column 'PE_ttm_Ratio' then return updated Merged_df """
        Merged_df['PE_ttm_Ratio'] = (Merged_df['Adj Close'] / Merged_df['EPS_ttm'])
        #st.write(Merged_df)
        return Merged_df

    def calc_volatility(ticker, OHLC_full_df):
        """Reads and edits the Original OHLC_full_df to calculate Adj Close_LogRet and Volatility then return updated OHLC_full_df """
        # create new column 'Adj Close_LogRet'
        OHLC_full_df['Adj Close_LogRet'] = np.log(OHLC_full_df['Adj Close'] / OHLC_full_df['Adj Close'].shift(1))
        OHLC_full_df['Volatility'] = OHLC_full_df['Adj Close_LogRet'].rolling(window=30,center=False).std() * np.sqrt(252)
        # st.write(OHLC_full_df)
        return OHLC_full_df

    def get_PE_stats(ticker,Merged_df):
        """Read in Merged_df and get PE stats returned by a dictionary """
        PE_stats = {}
        PE_stats["mean"] = Merged_df['PE_ttm_Ratio'].mean()
        PE_stats["std"] = Merged_df['PE_ttm_Ratio'].std()
        PE_stats["min"] = Merged_df['PE_ttm_Ratio'].min()
        PE_stats["max"] = Merged_df['PE_ttm_Ratio'].max()
        PE_stats["current"] = Merged_df['PE_ttm_Ratio'].iloc[-1]
        return PE_stats

    def get_volume_stats(ticker,OHLC_full_df):
        """Read in original OHLC_full_df and get volume stats returned by a dictionary """
        volume_stats = {}
        volume_stats["mean"] = OHLC_full_df['Volume'].mean()
        volume_stats["std"] = OHLC_full_df['Volume'].std()
        volume_stats["min"] = OHLC_full_df['Volume'].min()
        volume_stats["max"] = OHLC_full_df['Volume'].max()
        volume_stats["current"] = OHLC_full_df['Volume'].iloc[-1]
        return volume_stats

    def get_volatility_stats(ticker,OHLC_full_df):
        """Read in original OHLC_full_df and get volatility stats returned by a dictionary """
        volatility_stats = {}
        volatility_stats["mean"] = OHLC_full_df['Volatility'].mean()
        volatility_stats["std"] = OHLC_full_df['Volatility'].std()
        volatility_stats["min"] = OHLC_full_df['Volatility'].min()
        volatility_stats["max"] = OHLC_full_df['Volatility'].max()
        volatility_stats["current"] = OHLC_full_df['Volatility'].iloc[-1]
        return volatility_stats

    def normal_distribution_curve(dict_stats, label):
        """Draws the normal distribution curve """
        ## Grab from the dictionary
        mean = dict_stats["mean"]
        std = dict_stats["std"]
        x = dict_stats["current"]
        
        # for simulating the data 
        increment = 0.01

        # If it is volume we need to change the increments
        if label.lower() == "volume":
            increment = 1000
        else:
            increment = 0.01

        # Creating the distribution
        # start from the lowest 
        start = math.floor(mean - (3 * std))
        stop = math.ceil(mean + (3 * std))
        data = np.arange(start, stop + 1, increment)
        pdf = norm.pdf(data, loc=mean, scale=std)               # loc is the mean, scale is the standard deviation

        # Visualizing the distribution
        ## Adjust figure size for better viewing
        plt.figure(figsize=(16, 8))
        
        ## Plot the graph 
        plt.plot(data, pdf, color='black')
        
        ### Shade the corresponding area
        plt.fill_between(data, pdf, 0, where=(data <= x), color='#f59592')
        plt.fill_between(data, pdf, 0, where=(data > x), color='#97f4a6')
        #plt.axvline(x, color = '#383838')
        
        ## Correct labels
        plt.title('Probability Curve for {}'.format(ticker),fontsize=20)
        plt.xlabel(str(label))
        plt.ylabel('Probability Density')
        #plt.show()
        st.pyplot(plt)

    def get_CDF(dict_stats):
        """Grabs the CDF of one of the corresponding measurements (PE, Volume, Volatility) """
        mean = dict_stats["mean"]
        std = dict_stats["std"]
        current = dict_stats["current"]

        probability = round(norm.cdf(current, mean, std),4)
        return probability

    def suggestion(CDF,measure):
        """Gives suggestion based on its cdf and measure (PE,Volume,Volatility) """
        if measure == "PE":
            if CDF > .50:
                st.markdown('We suggest to **SELL** if within your risk.')
            elif CDF == .50:
                st.markdown('We suggest to **HOLD**.')
            else:
                st.markdown('We suggest to **BUY** if within your risk.')

        elif measure == "Volatility":
            if CDF > .50:
                st.markdown('**Sell Call** or **Buy Put** if within your risk')
            elif CDF == .50:
                st.markdown('We suggest to **HOLD**.')
            else:
                st.markdown('**Buy Call** or **Sell Put** if within your risk ')

        elif measure == "Volume":
            if CDF > .5:
                st.markdown('**Breakout** or **Breakdown**')
            elif CDF == .5:
                st.markdown('We suggest to **HOLD**.')
            else:
                st.markdown('**Capitulation Period**')

        else:
            st.error("NO MEASSURE SELECTED")

    if __name__ == '__main__':
        main()

        
with tab5:
    import streamlit as st
    import yfinance as yf
    import yahoo_fin.stock_info as si
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    import datetime as dt, matplotlib.pyplot as plt2, scipy.stats as scs

    st.title("VaR Monte Carlo Simulation")

    # s&p 500 tickers
    sp500_tickers = si.tickers_sp500()
    sp500_tickers.insert(0, 'Spy')
    sp500_tickers.insert(1, 'All')


    selected_option = st.multiselect(
        label='What are your favorite stocks',
        options=sp500_tickers,
        default=["Spy"],
        )

    if "All" in selected_option:
        selected_option = sp500_tickers[1:]

    time = st.slider(
        'Select Duration of Trading Days',
        1, 756, (45))
    st.write('time:', time)

    n_sims = st.number_input('Number of Simulation',min_value=100000, max_value=100000000, step=100000)
    st.write('Number of Simulation is: ', n_sims)


    for stock in selected_option:
        df = yf.download(stock, period='max')
        st.write(stock)

    # rename column names
    df.rename(columns = {'Open':'open', 'High':'high', 'Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'}, inplace = True)

    df['returns'] = df.adj_close.pct_change()
    df.dropna(inplace=True)

    s0 = df.adj_close[-1] # current stock price

    vol = df['returns'].std()*252**.5 # standard deviation or volatility

    # n_sims = 1000000 # number of simulation
    # Scrape Countries 10 years bond rate from website
    import pandas as pd
    countries_10_years_bond_rate = pd.read_html("http://www.worldgovernmentbonds.com/", header=0)[1]
    # filter usa 10 years bond data
    filter_usa_10years_bond = (countries_10_years_bond_rate['Unnamed: 1'] == 'United States')
    usa_10years_bond = countries_10_years_bond_rate[filter_usa_10years_bond]
    usa_10years_bond = usa_10years_bond.iloc[0]['10Y Bond'] 
    usa_10years_bond = float(usa_10years_bond.replace("%","")) /100
    rfr = usa_10years_bond

    # rfr = 0 # risk free rate
    # time = 45 # time period 45 days

    d = (rfr * 0.5 * vol**2) * (time/252)
    a = vol * np.sqrt(time/252)
    r = np.random.normal(0,1,(n_sims,1)) # random number 0 to 1 for 1 million simulations

    GBM_returns = s0 * np.exp(d + a*r) # Geometric Brownian Motion (GBM)

    # pers = [0.01, 0.1, 1.0, 2.5, 5.0, 10.0] # confidence interval
    confidence_interval = np.arange(0.01,0.11,0.01) # confidence interval

    var = scs.scoreatpercentile(GBM_returns -1, confidence_interval)

    df = pd.DataFrame(s0-var, confidence_interval, columns=['VaR'])
    st.table(df)

    plt2.hist(GBM_returns, density=True, bins=100)
    # plt.show()
    st.pyplot(plt2)

