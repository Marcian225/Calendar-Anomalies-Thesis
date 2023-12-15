import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import calendar
import datetime as dt
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import norm, shapiro
import arch 
import arch.unitroot as unitroot

def createReturnTable(file):
    # Read an input file and prepare lists for columns
    df = pd.read_csv(file, parse_dates=['Date'])
    pd.to_datetime(df['Date'], errors='raise')
    df.sort_values(by="Date", inplace=True)
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    dailyReturn = []
    dailyReturn.append(0)
    dailyLogReturn = []
    dailyLogReturn.append(0)
    dailyLogReturnNow = []
    dailyLogReturnNow.append(0)
    dayVar = []
    monthVar = []
    dateVar = df.index
    dateVar = [datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S") for timestamp in dateVar]

    # Calculate Logarithmic Return
    for index in range(1,len(df["Close"])):
        dailyLogReturn.append(pd.to_numeric(np.log(df["Close"].iloc[index,]))-pd.to_numeric(np.log(df["Close"].iloc[index-1,])))

    # Calculate Return
    for index in range(1,len(df["Close"])):
        dailyReturn.append(pd.to_numeric(df["Close"].iloc[index,])-pd.to_numeric(df["Close"].iloc[index-1,]))

    # Extract day-of-the-week and month-of-the-year for each observation
    for index in range(0, len(df.index)):
        dayVar.append(calendar.day_name[dateVar[index].weekday()])
        monthVar.append(calendar.month_name[dateVar[index].month])
 
    # Save to final dataframe
    portfolio = pd.DataFrame()
    portfolio['Date'] = dateVar
    portfolio['Return'] = dailyReturn
    portfolio['LogReturn'] = dailyLogReturn
    portfolio['Day'] = dayVar
    portfolio['Month'] = monthVar
    portfolio.set_index('Date', inplace=True)

    # Filter out extreme values
    lower_threshold = portfolio['Return'].quantile(0.01)
    upper_threshold = portfolio['Return'].quantile(0.99)
    portfolio = portfolio[(portfolio['Return'] >= lower_threshold) & (portfolio['Return'] <= upper_threshold)]

    return portfolio

def main():
    st.title('Calendar Effects')
    file = st.file_uploader("Upload CSV file", type="csv")

    if file is not None:
        portfolio = createReturnTable(file)
        detected_effects = []
        st.subheader("Data Structure")
        st.write("Below you can find a data structure created out of a file provided by you. It was preprocessed for further analysis to form containing 'Return', 'Month' and 'Day' columns")
        st.dataframe(portfolio, use_container_width=True)

        st.markdown('##')
        
        # Descriptive statistics table
        st.subheader("Descriptive Statistics")
        desc_stats = portfolio['LogReturn'].describe(include='all')
        desc_stats['Skewness'] = portfolio['LogReturn'].skew()
        desc_stats['Kurtosis'] = portfolio['LogReturn'].kurt()
        st.dataframe(desc_stats, use_container_width=True)

        # Generate insights based on the descriptive statistics
        mean_return = desc_stats['mean']
        std_deviation = desc_stats['std']
        skewness = desc_stats['Skewness']
        kurtosis = desc_stats['Kurtosis']
        resultsShapiro = shapiro(portfolio['LogReturn'])

        insights = []

        # Add shapiro normality test insights
        if resultsShapiro.pvalue > 0.05:
            insights.append(f"- Shapiro-Wilk Test: p-value is {resultsShapiro.pvalue}, returns are normally distributed")
        else:
            insights.append(f"- Shapiro-Wilk Test: p-value is {resultsShapiro.pvalue}, returns are not normally distributed")
        
        insights.append(f"- The mean return is {mean_return}, indicating the average return over the given period.")
        insights.append(f"- The standard deviation is {std_deviation}, representing the spread or volatility of returns.")

        # Add insights based on skewness and kurtosis
        if skewness > 0:
            insights.append(f"- Skewness: {skewness}. The return distribution is positively skewed, indicating a right tail.")
        elif skewness < 0:
            insights.append(f"- Skewness: {skewness}. The return distribution is negatively skewed, indicating a left tail.")
        else:
            insights.append(f"- Skewness: {skewness}. The return distribution is approximately symmetric.")

        if kurtosis > 0:
            insights.append(f"- Kurtosis: {kurtosis}. The return distribution has fat tails, indicating potential for extreme values.")
        elif kurtosis < 0:
            insights.append(f"- Kurtosis: {kurtosis}. The return distribution has thin tails, indicating a lack of extreme values.")
        else:
            insights.append(f"- Kurtosis: {kurtosis}. The return distribution has normal tails.")

        # Display insights
        st.write("Generated Insights:")
        for insight in insights:
            st.write(insight)
        st.markdown('##')

        sns.set_style("whitegrid")
        
        # # Plot of returns over time
        # returnPlot = plt.figure(figsize=(10,6))
        # sns.lineplot(data=portfolio, x=portfolio.index, y="Return", label="Return", legend = False, color="darkblue")
        # plt.title('Returns Over Time', fontsize=16)
        # plt.xlabel('', fontsize=12)
        # plt.ylabel('Return', fontsize=12)
        # st.pyplot(returnPlot)

        # st.markdown('##')

        # Plot of log returns over time
        logReturnPlot = plt.figure(figsize=(10,6))
        sns.lineplot(data=portfolio, x=portfolio.index, y="LogReturn", label="LogReturn", legend = False, color="darkblue")
        plt.title('Logarithmic Returns Over Time', fontsize=16)
        plt.xlabel('', fontsize=12)
        plt.ylabel('Return', fontsize=12)
        st.pyplot(logReturnPlot)

        st.markdown('##')


        norm_axis = np.linspace(min(portfolio['LogReturn']), max(portfolio['LogReturn']), 1000)

        # Plot of returns distribution
        distributionPlot = plt.figure(figsize=(10, 6))
        sns.histplot(portfolio['LogReturn'], edgecolor='black', kde=False, color="darkblue", stat="density")
        sns.lineplot(x=norm_axis, y=norm.pdf(norm_axis, mean_return, std_deviation), label='Normal Distribution', color='red', linewidth=3)
        plt.title('Distribution of Logarithmic Returns', fontsize=16)
        plt.xlabel('Log Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True)
        st.pyplot(distributionPlot)

        st.markdown('##')

        # Data stationarity

        # Augmented Dickey-Fuller Test
        resultsADF = unitroot.ADF(portfolio.loc[:,'LogReturn'])
        st.subheader("Augmented Dickey-Fuller Test Result")
        st.write("ADF Statistic:", resultsADF.stat)
        st.write("p-value:", resultsADF.pvalue)
        if resultsADF.pvalue < 0.05:
            st.write("- ADF Test: The data is likely stationary.")
        else:
            st.write("- ADF Test: The data is likely non-stationary.")
        st.markdown('##')

        # Phillips-Perron Test
        resultsPP = unitroot.PhillipsPerron(portfolio.loc[:,'LogReturn'])
        st.subheader("Phillips-Perron Test Result")
        st.write("PP Statistic:", resultsPP.stat)
        st.write("p-value:", resultsPP.pvalue)
        if resultsPP.pvalue < 0.05:
            st.write("- Phillips-Perron Test: The data is likely stationary.")
        else:
            st.write("- Phillips-Perron Test: The data is likely non-stationary.")
        st.markdown('##')

        # Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test
        resultsKPSS = unitroot.KPSS(portfolio.loc[:,'LogReturn'])
        st.subheader("Kwiatkowski–Phillips–Schmidt–Shin Test Result")
        st.write("KPSS Statistic:", resultsKPSS.stat)
        st.write("p-value:", resultsKPSS.pvalue)
        if resultsKPSS.pvalue < 0.05:
            st.write("- KPSS Test: The data is likely non-stationary around a deterministic trend.")
        else:
            st.write("- KPSS Test: The data is likely stationary.")
        st.markdown('##')

        # OLS regression model for days
        dummyDayPortfolio = pd.get_dummies(portfolio, columns=['Day'])
        X = dummyDayPortfolio[['Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 'Day_Monday', 'Day_Friday']]
        dayOlsModel = sm.OLS(dummyDayPortfolio['Return'], X).fit()
        st.subheader("OLS Regression Results for Days")
        st.write(dayOlsModel.summary())
        st.markdown('##')

        for day in dummyDayPortfolio[['Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 'Day_Monday', 'Day_Friday']]:
            if dayOlsModel.pvalues.loc[day] < 0.05:
                st.write(f"- Day-of-the-week Effect (OLS): {day} - Returns tend to be different on {day}.")
        st.markdown('##')

        # OLS regression model for months
        dummyMonthPortfolio = pd.get_dummies(portfolio, columns=['Month'])
        X = dummyMonthPortfolio[['Month_April', 'Month_August', 'Month_December',
            'Month_February', 'Month_January', 'Month_July', 'Month_June',
            'Month_March', 'Month_May', 'Month_November', 'Month_October',
            'Month_September']]
        monthOlsModel = sm.OLS(dummyMonthPortfolio['Return'], X).fit()
        st.subheader("OLS Regression Results for Months")
        st.write(monthOlsModel.summary())
        st.markdown('##')

        for month in dummyMonthPortfolio[['Month_April', 'Month_August', 'Month_December',
                    'Month_February', 'Month_January', 'Month_July', 'Month_June',
                    'Month_March', 'Month_May', 'Month_November', 'Month_October',
                    'Month_September']]:
            if monthOlsModel.pvalues.loc[month] < 0.05:
                st.write(f"- Month-of-the-year Effect (OLS): {month} - Returns tend to be different on {month}.")
        st.markdown('##')

        # Average month-of-the-year/day-of-the-week return
        avgDayReturn = portfolio.groupby('Day')['Return'].mean()
        avgMonthReturn = portfolio.groupby('Month')['Return'].mean()

        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Plot for average day returns
        avgDayPlot = plt.figure(figsize=(12, 6))
        day_colors = ['black' if pval >= 0.05 else 'red' for pval in dayOlsModel.pvalues]
        sns.barplot(x=avgDayReturn.index, y=avgDayReturn.values, order=day_order, palette=day_colors)
        #sns.barplot(x=avgDayReturn.index, y=avgDayReturn.values, order = day_order, palette="viridis")
        plt.title('Average Daily Returns', fontsize=16)
        plt.xlabel('', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.legend(handles=[plt.Line2D([0], [0], color='black', lw=4, label='Insignificant'),
                            plt.Line2D([0], [0], color='red', lw=4, label='Significant')],
                            title='Statistical Significance')
        plt.grid(True)
        st.pyplot(avgDayPlot)

        st.markdown('##')

        # Plot for average month returns
        avgMonthPlot = plt.figure(figsize=(12, 6))
        month_colors = ['black' if pval >= 0.05 else 'red' for pval in monthOlsModel.pvalues]
        sns.barplot(x=avgMonthReturn.index, y=avgMonthReturn.values, order=month_order, palette=month_colors)
        plt.title('Average Monthly Returns', fontsize=16)
        plt.xlabel('', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.legend(handles=[plt.Line2D([0], [0], color='black', lw=4, label='Insignificant'),
                            plt.Line2D([0], [0], color='red', lw=4, label='Significant')],
                            title='Statistical Significance')
        plt.grid(True)
        st.pyplot(avgMonthPlot)

        st.markdown('##')
        st.subheader("EGARCH Modeling Results")

        # EGARCH Modelling and its summary
        model_egarch = arch.arch_model(portfolio['LogReturn'], vol='EGARCH', p=1, q=1, rescale=True)
        results_egarch = model_egarch.fit()
        portfolio['Volatility'] = results_egarch.conditional_volatility
        st.write(results_egarch.summary())
        st.dataframe(portfolio)

        # Mean daily volatility
        dailyVolatility = pd.DataFrame()
        dailyVolatility['Avg Volatility'] = portfolio.groupby('Day')['Volatility'].mean()
        dailyVolatilityPlot = plt.figure(figsize=(10, 6))
        sns.barplot(data = dailyVolatility, x = dailyVolatility.index, y="Avg Volatility", color = "darkblue", order=day_order)
        plt.title('Average Day-of-the-week Volatility (EGARCH Model)', fontsize = 16)
        plt.xlabel('')
        plt.ylabel('Avg Volatility', fontsize=12)
        st.dataframe(dailyVolatility, use_container_width=True)
        st.pyplot(dailyVolatilityPlot)

        # Mean monthly volatility
        monthlyVolatility = pd.DataFrame()
        monthlyVolatility['Avg Volatility'] = portfolio.groupby('Month')['Volatility'].mean()
        monthlyVolatilityPlot = plt.figure(figsize=(10, 6))
        sns.barplot(data = monthlyVolatility, x = monthlyVolatility.index, y="Avg Volatility", color = "darkblue", order=month_order)
        plt.title('Average Month-of-the-year Volatility (EGARCH Model)', fontsize = 16)
        plt.xlabel('')
        plt.ylabel('Avg Volatility', fontsize=12)
        st.dataframe(monthlyVolatility, use_container_width=True)
        st.pyplot(monthlyVolatilityPlot)

        # Volatility over time
        volatilityPlot = plt.figure(figsize=(10, 6))
        sns.lineplot(x=results_egarch._index, y=results_egarch.conditional_volatility, label='Volatility', legend = False, color = "darkblue")
        plt.title('Volatility Over Time (EGARCH Model)', fontsize = 16)
        plt.xlabel('')
        plt.ylabel('Volatility', fontsize=12)
        st.pyplot(volatilityPlot)

# Run the app
if __name__ == "__main__":
    main()