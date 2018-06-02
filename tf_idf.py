from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from nltk.corpus import stopwords

from pdb import set_trace as bp

class TfIdf:

    def __init__(self, corpus):
        self.corpus = corpus

    def compute_scores(self):
        self.sklearn_tfidf = TfidfVectorizer()
        self.scores = self.sklearn_tfidf.fit_transform(self.corpus)

    def vocabulary(self):
        return self.sklearn_tfidf.vocabulary_

    def inverted_vocabulary(self):
        result = {}
        for key, values in self.vocabulary().items():
            result[values] = key
        return result

    def remove_small_length_terms(self, length):
        self.removed_terms_corpus = []
        for document in self.corpus:
            terms_to_remove = []
            document_terms = document.split()
            for word in document_terms:
                if len(word) <= length:
                    terms_to_remove.append(word)

            for term_to_remove in terms_to_remove:
                try:
                    document_terms.remove(term_to_remove)
                except Exception:
                    pass

            new_document = ' '.join(document_terms)
            self.removed_terms_corpus.append(new_document)

    def remove_stop_words(self):
        stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herse"', 'him', 'himse"', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itse"', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myse"', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']
        self.removed_terms_corpus = []
        for document in self.corpus:
            terms_to_remove = []
            document_terms = document.split()
            for word in document_terms:
                if word in stop_words:
                    terms_to_remove.append(word)

            for term_to_remove in terms_to_remove:
                try:
                    document_terms.remove(term_to_remove)
                except Exception:
                    pass

            new_document = ' '.join(document_terms)
            self.removed_terms_corpus.append(new_document)

    def remove_terms(self, threshold):

        inverted_vocabulary = self.inverted_vocabulary()

        self.removed_terms_corpus = []
        for document_idx, document in enumerate(self.corpus):
            # print(document)
            document_terms = document.split()
            terms_to_remove = []
            for indice_to_analize in self.scores[document_idx].indices:
                # print(inverted_vocabulary[indice_to_analize], self.scores[document_idx, indice_to_analize], threshold)
                if self.scores[document_idx, indice_to_analize] < threshold:
                    terms_to_remove.append(inverted_vocabulary[indice_to_analize])
                    # print(terms_to_remove)

            for term_to_remove in terms_to_remove:
                try:
                    document_terms.remove(term_to_remove)
                except Exception:
                    pass

            new_document = ' '.join(document_terms)
            self.removed_terms_corpus.append(new_document)
            # print(new_document)
            # print()
            # print(document, '|', terms_to_remove, '|', self.removed_terms_corpus[-1])

    def vocabulary_and_scores(self):
        inverted_vocabulary = self.inverted_vocabulary()

        self.document_terms_and_scores = {}
        for document_idx, document in enumerate(self.corpus):
            terms_and_score = {}
            document_terms = document.split()
            for indice_to_analize in self.scores[document_idx].indices:
                terms_and_score[inverted_vocabulary[indice_to_analize]] = self.scores[document_idx, indice_to_analize]

            self.document_terms_and_scores[document_idx] = terms_and_score

    def most_frequent_terms(self, n):
        terms_and_frequencies = {}
        terms_and_frequencies = defaultdict(lambda:0, terms_and_frequencies)
        for document in self.corpus:
            for word in document.split():
                terms_and_frequencies[word] += 1

        stats = Counter(terms_and_frequencies)
        return stats.most_common(n)
