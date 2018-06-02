from dataset import *
from word_vectorizer import *
from tf_idf import *
from n_grams import *
from principal_components import *
from single_value_decomposition import *
from k_means_cluster import *
from pdb import set_trace as bp
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

filename = 'dev_dataset.csv'
ds = Dataset(filename)
logging.info('Loading dataset')
ds.load()
logging.info('Dataset loaded with ' + str(ds.sample_count()) + ' samples')

logging.info('Removing duplicates')
ds.remove_duplicates()
logging.info('Duplicates removed. Dataset has now ' + str(ds.sample_count()) + ' samples')

corpus = ds.headline_texts()

tf_idf = TfIdf(corpus)
logging.info('Computing TfIdf scores')
tf_idf.compute_scores()
logging.info('TfIdf scores computed')

logging.info('Removing non-relevant terms')
tf_idf.remove_terms(0.3)
logging.info('Non-relevant terms removed')

corpus_words = []
removed_terms_corpus = tf_idf.removed_terms_corpus
for sentence in removed_terms_corpus:
    corpus_words.append(sentence.split())

word_vectorizer = WordVectorizer(corpus_words)
logging.info('Computing word2vec')
word_vectorizer.compute_scores()
logging.info('Word2vec computed')
# word_vectorizer.draw()

n_clusters = 50
X = word_vectorizer.model[word_vectorizer.model.wv.vocab]
k_means_cluster = KMeansCluster(X)
logging.info('Computing K-Means clusters')
k_means_cluster.fit_transform(n_clusters)

from collections import Counter, defaultdict

final_document_clusters = {}
final_document_clusters = defaultdict(lambda:[], final_document_clusters)

for document in removed_terms_corpus:
    document_clusters = []
    for word in document.split():
        try:
            vocab_index = word_vectorizer.model.wv.vocab[word].index
            document_clusters.append(k_means_cluster.labels()[vocab_index])
        except Exception:
            pass
    counter = Counter(document_clusters)
    final_cluster = max(counter, key = lambda key: counter[key])
    final_document_clusters[final_cluster].append(document)
    # print(document)
    # print(final_document_clusters)

bp()
counter = Counter(final_document_clusters)



for cluster in final_document_clusters.keys():
    for document in final_document_clusters[cluster]:
        print(document)
    print()



# k_means_cluster = KMeansCluster(word_vectorizer.model[word_vectorizer.model.wv.vocab])
# logging.info('Computing K-Means clusters')
# k_means_cluster.fit_transform(n_clusters)
# for cluster_number in range(n_clusters):
#     for document_indice in k_means_cluster.cluster(cluster_number):
#         print(corpus[document_indice])
#     print()

#
# corpus = ds.headline_texts()
#
#
# # n_clusters = 500
# # k_means_cluster = KMeansCluster(tf_idf.scores)
# # k_means_cluster.fit_transform(n_clusters)
# # for cluster_number in range(n_clusters):
# #     for document_indice in k_means_cluster.cluster(cluster_number):
# #         print(corpus[document_indice])
# #     print()
#

#
# n_grams = NGrams(removed_terms_corpus)
# logging.info('Producing 2-term grams')
# n_grams.produce_word_grams(2)
# logging.info('Producing 4-char grams')
# n_grams.produce_char_grams(4)
#
# logging.info('Merging term/char-grams into corpus')
# expanded_corpus = []
# for idx, document in enumerate(removed_terms_corpus):
#     expanded_corpus.append(
#                             document + ' ' + \
#                             n_grams.word_n_grams[idx] + ' ' + \
#                             n_grams.char_n_grams[idx] \
#                             )
#
# tf_idf = TfIdf(expanded_corpus)
# logging.info('Computing TfIdf scores for expanded corpus')
# tf_idf.compute_scores()
# scores = tf_idf.scores
#
# # n_clusters = 500
# # k_means_cluster = KMeansCluster(tf_idf.scores)
# # k_means_cluster.fit_transform(n_clusters)
# # for cluster_number in range(n_clusters):
# #     for document_indice in k_means_cluster.cluster(cluster_number):
# #         print(corpus[document_indice])
# #     print()
#
# # svd = SingleValueDecomposition(scores)
# # logging.info('Computing SVD')
# # svd.fit_transform()
#
# k_means_cluster = KMeansCluster(scores)
#
# # n_clusters = 500
# # logging.info('Computing K-Means clusters')
# # k_means_cluster.fit_transform(n_clusters)
# # for cluster_number in range(n_clusters):
# #     for document_indice in k_means_cluster.cluster(cluster_number):
# #         print(corpus[document_indice])
# #     print()
#
# logging.info('Computing range of K-Means clusters')
# k_means_cluster.fit_transform_range(1, 1000, 10)
# print(k_means_cluster.cost)
# print(k_means_cluster.distortions)
#
# # n_clusters = 500
# # k_means_cluster.fit_transform(n_clusters)
# # for cluster_number in range(n_clusters):
# #     for document_indice in k_means_cluster.cluster(cluster_number):
# #         print(removed_terms_corpus[document_indice])
# #     print()
