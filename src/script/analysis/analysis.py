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
    
    def drop_data(self):
        self.droped = self.data.dropna(how='all')
    
    def select_cut(self, cut=.9):
        count = cut * len(self.droped)
        self.cleaned_data = self.droped.dropna(axis=1, thresh=count)
        self.cleaned_data.to_excel('src/data/cleaned.xlsx')

analysis = Analysis()
analysis.drop_data()
analysis.select_cut()
