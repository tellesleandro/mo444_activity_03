from sklearn.decomposition import TruncatedSVD

from pdb import set_trace as bp

class SingleValueDecomposition:

    def __init__(self, X):
        self.X = X;

    def fit_transform(self):
        self.svd = TruncatedSVD(n_components = 30)
        self.principal_components = self.svd.fit_transform(self.X.toarray())
