import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

db = pd.read_csv("../data_sets/medical_charge.csv")
db.smoker = db.smoker.map({"yes": 1, "no": 0})
print(db)

print("데이터 정보 : \n{}\n".format(db.info()))
print("=================================\n")
print("데이터 크기 : \n{}\n".format(db.shape))
print("=================================\n")
print("데이터 Overview : \n{}\n".format(db.describe()))
print("=================================\n")

stacked_data = db.groupby(["smoker", "sex"]).size().reset_index().pivot(columns="sex", index="smoker", values=0)
print(stacked_data)

plt.figure(figsize=(15, 30))

sns.factorplot(kind="bar", x="region", hue="sex", y="bmi", data=db)
plt.show()
