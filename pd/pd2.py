import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("~/Downloads/query2.csv")
# print(df)

df['日期'].plot()
plt.show()