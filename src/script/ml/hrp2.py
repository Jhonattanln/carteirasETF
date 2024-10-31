import numpy as np
import cudf
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns

class HierarchicalRiskParity:
    def __init__(self, returns):
        self.returns = returns
        self.corr = returns.corr()
        self.cov = returns.cov()
        self.dist = self._get_correlation_distance()
        self.link = None
        self.sort_ix = None
        self.weights = None

    def _get_correlation_distance(self):
        return np.sqrt((1 - self.corr) / 2)

    def _get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = cudf.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = cudf.Series(link[j, 1], index=i+1)
            sort_ix = cudf.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _get_cluster_variance(self, cluster_items):
        cov_slice = self.cov.loc[cluster_items, cluster_items]
        weights = np.linalg.inv(cov_slice).sum(axis=1)
        weights /= weights.sum()
        return np.dot(np.dot(weights.T, cov_slice), weights)

    def _get_recursive_bisection(self, sort_ix):
        weights = cudf.Series(1, index=sort_ix)
        clusters = [sort_ix]
        while len(clusters) > 0:
            clusters = [cluster[j:k] for cluster in clusters
                        for j, k in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                        if len(cluster) > 1]
            for i in range(0, len(clusters), 2):
                cluster0 = clusters[i]
                cluster1 = clusters[i + 1]
                cluster0_variance = self._get_cluster_variance(cluster0)
                cluster1_variance = self._get_cluster_variance(cluster1)
                alpha = 1 - cluster0_variance / (cluster0_variance + cluster1_variance)
                weights[cluster0] *= alpha
                weights[cluster1] *= (1 - alpha)
        return weights

    def compute_hrp(self):
        self.link = hierarchy.linkage(squareform(self.dist.to_numpy()), 'single')
        self.sort_ix = self._get_quasi_diag(self.link)
        self.sort_ix = self.corr.index[self.sort_ix].tolist()
        self.weights = self._get_recursive_bisection(self.sort_ix)
        self.weights.to_csv('src/data/weights.csv')
        return self.weights

    def plot_corr_matrix(self, title, corr=None):
        if corr is None:
            corr = self.corr
        plt.figure(figsize=(15, 8))
        sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('images/hrp_matrix.png')

    def plot_dendrogram(self):
        plt.figure(figsize=(10, 8))
        hierarchy.dendrogram(self.link, labels=self.returns.columns)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Asset')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig('images/dendogram.png')

    def plot_hrp_weights(self):
        plt.figure(figsize=(15, 8))
        self.weights.sort_values(ascending=True).plot(kind='bar')
        plt.title('HRP Portfolio Weights')
        plt.xlabel('Asset')
        plt.ylabel('Weight')
        plt.tight_layout()
        plt.savefig('images/hrp_weights.png')

    def get_portfolio_stats(self):
        portfolio_return = (self.returns * self.weights).sum(axis=1)
        expected_return = portfolio_return.mean() * 252 
        volatility = portfolio_return.std() * np.sqrt(252)
        risk_free_rate = 0.11
        return {
            'Expected Annual Return': expected_return,
            'Annual Volatility': volatility,
            'Sharpe Ratio': (expected_return - risk_free_rate) / volatility
        }

# Example usage
if __name__ == "__main__":
    # Generate some random return data
    np.random.seed(42)
    returns = cudf.read_csv('src/data/returns.csv', sep=',', index_col=0)

    # Create HRP object and compute weights
    hrp = HierarchicalRiskParity(returns)
    weights = hrp.compute_hrp()

    # Plot visualizations
    hrp.plot_corr_matrix('Correlation Matrix of Asset Returns')
    hrp.plot_dendrogram()
    hrp.plot_corr_matrix('Sorted Correlation Matrix', hrp.corr.loc[hrp.sort_ix, hrp.sort_ix])
    hrp.plot_hrp_weights()

    # Print results
    print("HRP Portfolio Weights:")
    print(weights)
    print(f"\nSum of weights: {weights.sum():.4f}")

    stats = hrp.get_portfolio_stats()
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")