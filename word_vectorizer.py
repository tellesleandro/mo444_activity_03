from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from pdb import set_trace as bp

class WordVectorizer:

    def __init__(self, corpus):
        self.corpus = corpus

    def compute_scores(self):
        self.model = Word2Vec(self.corpus, min_count = 1)

    def vocabulary(self):
        return list(model.wv.vocab)

    def draw(self):
        X = self.model[self.model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        plt.scatter(result[:, 0], result[:, 1])
        words = list(self.model.wv.vocab)
        for i, word in enumerate(words):
        	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        plt.show()
