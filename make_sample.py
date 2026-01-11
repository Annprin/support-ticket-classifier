import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=["Churn"])
df.head(5).to_csv("sample.csv", index=False)
print("saved sample.csv")