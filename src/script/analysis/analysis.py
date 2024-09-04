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
        