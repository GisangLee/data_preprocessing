import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("../data_sets/titanic.xls")
print(data.info())
print(data.describe())

