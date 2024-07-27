import pandas as pd

path = './SalesData.csv'
data = pd.read_csv(path)

table = pd.crosstab(
    [data.Year, data.Client],
    data.Quarter,
    values=data.Revenue,
    aggfunc="sum",
    margins=True
)

table.reset_index(inplace=True)

print(table)
