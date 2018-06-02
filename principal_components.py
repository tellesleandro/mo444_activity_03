from sklearn.decomposition import PCA

from pdb import set_trace as bp

class PrincipalComponents:

    def __init__(self, X):
        self.X = X;

    def fit_transform(self):
        self.pca = PCA()
        self.principal_components = self.pca.fit_transform(self.X.toarray())
