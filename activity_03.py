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

filename = 'dataset_2016.csv'
ds = Dataset(filename)
logging.info('Loading dataset')
ds.load()
logging.info('Dataset loaded with ' + str(ds.sample_count()) + ' samples')

logging.info('Removing duplicates')
ds.remove_duplicates()
logging.info('Duplicates removed. Dataset has now ' + str(ds.sample_count()) + ' samples')

logging.info('Saving word-cloud file')
ds.save_word_cloud()
logging.info('Word-cloud file saved')

corpus = ds.headline_texts()

tf_idf = TfIdf(corpus)
logging.info('Removing small-length terms')
tf_idf.remove_small_length_terms(3)
logging.info('Small-length terms removed')

tf_idf = TfIdf(tf_idf.removed_terms_corpus)
logging.info('Removing small-length terms')
tf_idf.remove_small_length_terms(3)
logging.info('Small-length terms removed')

tf_idf = TfIdf(tf_idf.removed_terms_corpus)
logging.info('Removing stop words')
tf_idf.remove_stop_words()
logging.info('Stop words removed')

tf_idf = TfIdf(tf_idf.removed_terms_corpus)
logging.info('Computing TfIdf scores')
tf_idf.compute_scores()
logging.info('TfIdf scores computed')
logging.info('Removing non-relevant terms')
tf_idf.remove_terms(0.3)
logging.info('Non-relevant terms removed')

tf_idf = TfIdf(tf_idf.removed_terms_corpus)
logging.info('Computing 15-most frequent words')
most_frequent_terms = tf_idf.most_frequent_terms(15)
logging.info('15-most frequent words computed')
print(most_frequent_terms)

n_grams = NGrams(tf_idf.corpus)
logging.info('Producing stemmed documents')
n_grams.produce_stem()
logging.info('Stemmed documents produced')

logging.info('Computing 15-most frequent stem-word')
most_frequent_stem_word = n_grams.most_frequent_stem_word(15)
logging.info('15-most frequent stem-word computed')
print(most_frequent_stem_word)

n_grams = NGrams(n_grams.stemmed)
logging.info('Producing 2-term grams')
n_grams.produce_word_grams(2)

logging.info('Computing 15-most frequent word-gram')
most_frequent_word_grams = n_grams.most_frequent_word_grams(15)
logging.info('15-most frequent word-gram computed')
print(most_frequent_word_grams)

logging.info('Merging term-grams/stemm into corpus')
expanded_corpus = []
for idx, document in enumerate(n_grams.corpus):
    expanded_corpus.append(
                            document + ' ' + \
                            n_grams.word_n_grams[idx] \
                            )
logging.info('Corpus merged')

tf_idf = TfIdf(expanded_corpus)
logging.info('Computing TfIdf scores for expanded corpus')
tf_idf.compute_scores()
scores = tf_idf.scores

# svd = SingleValueDecomposition(scores)
# logging.info('Computing SVD')
# svd.fit_transform()
# logging.info('SVD computed')

n_clusters = 200
k_means_cluster = KMeansCluster(scores)
k_means_cluster.fit_transform(n_clusters)
logging.info('K-Means clusters computed')
for cluster_number in range(n_clusters):
    for document_indice in k_means_cluster.cluster(cluster_number):
        print(corpus[document_indice])
    print()

# k_means_cluster = KMeansCluster(scores)
# logging.info('Computing range of K-Means clusters')
# k_means_cluster.fit_transform_range(1, 200, 1)
# print(k_means_cluster.cost)
# print(k_means_cluster.distortions)
