import pandas_datareader.data as web
import pandas as pd


def getData():
    all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ['AAPL', "IBM", "MSFT", "GOOG"]}
    price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
    volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
    returns = price.pct_change()
    dict_of_df = {"volume": volume, "returns": returns}
    return dict_of_df


def corr_cov(df):
    print("상관관계 : \n{}\n".format(df['returns'].corr()))
    print("공분산: \n{}\n".format(df['returns'].cov()))
    print("Corr With: \n{}\n".format(df['returns'].corrwith(df['returns'].IBM)))
    print("Corr Dataframe: \n{}\n".format(df['returns'].corrwith(df['volume'])))


result = getData()
print(result)
corr_cov(result)
