import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data_sets/movies.csv")


def get_data(data):
    print("데이터 정보 : \n", data.info())
    print("데이터 NULL 합계 : ", data.isnull().sum())
    print("데이터 top 5\n")
    print(data.head())
    print("데이터 서머리 : \n")
    print(data.describe())
    print("변수 : \n")
    print(data.columns)


data = data.drop(['homepage'], axis=1)

print(data[['popularity', 'vote_average', 'production_companies']])

ax = sns.barplot(x='vote_average', y='title', data=data.head(50))
ax.plot()
plt.show()

get_data(data)
