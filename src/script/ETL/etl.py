#%% Import data from excle file
import pandas as pd

# Read data from excel file
df = pd.read_excel('src/data/etf.xlsx', parse_dates=True, index_col=0, skiprows=3)
df.head()
# %%