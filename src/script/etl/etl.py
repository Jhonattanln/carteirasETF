# Import data from excle file
import datetime
import numpy as np
import pandas as pd

# Trasform data class
class TransformData:
    """
    Class to trasform financial data
    - Rename columns
    - Slice data
    - Clean data
    """

    def __init__(self):
        self.etf = None

    def transform_data(self, file_path):
        "Funtion to transform data"
        self.etf = pd.read_excel(file_path, parse_dates=True, index_col=0, skiprows=3)
        self.etf.columns = self.etf.columns.str[-6:]
        data = datetime.datetime.today() - datetime.timedelta(days=365)
        self.etf = self.etf.loc[data:datetime.datetime.today().strftime('%Y-%m-%d')]
        self.etf.replace('-', np.nan, inplace=True)
        return self.etf

    def load_data(self):
        "Function to load data"
        self.etf.to_excel('src/data/transformed_data.xlsx')

transformer = TransformData()
transformed_data = transformer.transform_data('src/data/etf.xlsx')
transformer.load_data()
