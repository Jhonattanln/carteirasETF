"""
Scrip para analise de dados
- Contagem de valores nulos
- Correlação
- Extração
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis:
    """
    Class to data analysis
    """
    def __init__(self):
        self.data = pd.read_excel('src/data/transformed_data.xlsx', parse_dates=True, index_col=0)
        self.droped = None
        self.cleaned_data = None
    
    def manipulate_data(self, cut=0.9):
        self.droped = self.data.dropna(how='all')
        count = cut * len(self.droped)
        self.droped = self.droped.dropna(how='all', thresh=count)
        return self.droped

    def export_excel(self, file_path):
        self.droped.to_excel(file_path)
        return print('Arquivo exportado!')

analysis = Analysis()
analysis.drop_data()
analysis.select_cut()
