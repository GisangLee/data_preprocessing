import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../data_sets/Salaries.csv")
pre_data = data[['JobTitle', 'BasePay', 'OvertimePay', 'OtherPay', 'TotalPay', 'TotalPayBenefits', 'Year']]

encoder = LabelEncoder()
encoder.fit(pre_data.JobTitle)
labeled_data = encoder.transform(pre_data.JobTitle).reshape(-1, 1)
print("인코딩 : {}".format(labeled_data))

pre_data = pd.concat([pre_data, pd.DataFrame(labeled_data)], axis=1)
pre_data.rename(columns={0: 'JobEncoding'}, inplace=True)


def to_float(col):
    if "Not Provided" in col:
        return 0


pre_data['BasePay'] = pre_data.apply(lambda x: to_float(pre_data['BasePay']), axis=1)
pre_data['OvertimePay'] = pre_data.apply(lambda x: to_float(pre_data['OvertimePay']), axis=1)
pre_data['OtherPay'] = pre_data.apply(lambda x: to_float(pre_data['OtherPay']), axis=1)

pre_data['BasePay'] = pre_data['BasePay'].astype(float)
pre_data['OvertimePay'] = pre_data['OvertimePay'].astype(float)
pre_data['OtherPay'] = pre_data['OtherPay'].astype(float)
pre_data['Year'] = pre_data['Year'].astype(int)

data_2011 = pre_data[pre_data.Year == 2011]
data_2012 = pre_data[pre_data.Year == 2012]
data_2013 = pre_data[pre_data.Year == 2013]
data_2014 = pre_data[pre_data.Year == 2014]

print(pre_data.describe())
print(pre_data.head())
print(pre_data.isnull())
print(data_2011.head())
print(data_2012.head())
print(data_2013.head())
print(data_2014.head())
print(data_2011.isnull())
