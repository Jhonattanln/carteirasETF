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
        self.data = pd.read_csv('src/data/transformed_data.csv', parse_dates=True, index_col=0)
        self.droped = None
        self.corr = None

    def manipulate_data(self, cut=0.9):
        self.droped = self.data.dropna(how='all')
        count = cut * len(self.droped)
        self.droped = self.droped.dropna(axis=1, thresh=count)
        self.droped = self.droped.ffill()
        self.droped = self.droped.pct_change().dropna()
        self.droped = self.droped.drop('LFTS11', axis=1)
        self.droped.to_csv('src/data/returns.csv')

        return self.droped

    def correl_data(self):
        self.corr = self.droped.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlação entre ETFs')
        plt.savefig('images/correlation.png')

    def export_excel(self, file_path):
        self.droped.to_csv(file_path)
        return print('Arquivo exportado!')

analysis = Analysis()
analysis.manipulate_data()
analysis.correl_data()
analysis.export_excel('src/data/etf.csv')
