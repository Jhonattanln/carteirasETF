# Import data from excle file
import datetime
import numpy as np
import pandas as pd

# Read data from excel file
df = pd.read_excel('src/data/etf.xlsx', parse_dates=True, index_col=0, skiprows=3)

## Transform data
# Rename columns
df.columns = df.columns.str[-6:]

# Slice data
data = datetime.datetime.today() - datetime.timedelta(days=365)
df = df.loc[data:datetime.datetime.today().strftime('%Y-%m-%d')]

# Replace data
df.replace('-', np.nan, inplace=True)