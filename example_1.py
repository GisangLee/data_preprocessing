import pandas as pd

data = pd.read_csv("data_sets/heart_disease_db.csv")


def get_data(data):
    print("데이터 : ", data)
    print("데이터 : ", data.shape)
    print("데이터 컬럼 : ", list(data.columns))


get_data(data)
