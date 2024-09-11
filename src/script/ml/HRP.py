import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd

# Hierarichal Risk Parity

def getIVP(cov, **kargs):
    ivp = 1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    cov_=cov.loc[cItems, cItems]
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_) [0, 0]
    return cVar

def getQuasiDiag(lik):
    link = link.astype(int)
    sortIx = pd.Series(link[-1, 0], link[-1, 1])
    numItems=link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values-numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i+1)
        sortIx = sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        for i in xrange(0, len(cItems), 2):
            cItems0 = cItems[i] #cluster 1
            cItems1 = cItems[i+1] # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha # weigh 1
            w[cItems1] *= alpha #weigh 2
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0 <= d[i, j] <= 1
    # This is a proper distance metric
    dist = ((1 - corr) / 2) ** .5
    return dist

def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of correlation matrix
    if label is None: labels=[]
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.range(.5, corr.shape[0] + .5), labels)
    plt.xticks(np.range(.5, corr.shape[0] + .5), labels)
    plt.savefig(path)
    plt.clf()
    plt.close()
    return

def generateData(nObs, size0, size1, sigma1):
    # Time series of correlated variables
    # Create an uncorrelated data
    np.random.seed(42)
    random.seed(42)
    x = np.random.normal(0, 1, size=(nObs, size0)) # each row is a variable

    # Create a correlated data
    cols = [random.randint(0, size1) for i in xrange(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols

def main():
    # Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, .25
    x, cols = generateData(nObs, size0, size1, sigma1)
    print([(j + 1, size0 + i) for i, j in enumerate(cols, 1)])
    cov, corr = x.cov(), x.corr()

    # Compute and plot correl matrix
    plotCorrMatrix('images/HRP_corr0.png', corr, labels=corr.columns)

    # Cluster
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx, sortIx].tolist() # recover labels
    df0 = corr.loc[sortIx, sortIx] # reorder
    plotCorrMatrix('images/HRp_corr1.png', df0, labels=df0.columns)

    # Capital Allocation
    hrp = getRecBipart(cov, sortIx)
    print(hrp)
    return

if __name__=='__main__':main()
