from nltk import ngrams
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict

from pdb import set_trace as bp

class NGrams:

    def __init__(self, corpus):
        self.corpus = corpus

    def produce_word_grams(self, n):
        self.word_grams_dictionary = {}
        self.word_grams_dictionary = defaultdict(lambda:0, self.word_grams_dictionary)
        self.word_n_grams = []
        for document in self.corpus:
            joined_n_grams = ''
            n_grams_sentences = ngrams(document.split(), n)
            for n_grams_sentence in n_grams_sentences:
                new_gram = '_'.join(n_grams_sentence)
                joined_n_grams += new_gram + ' '
                self.word_grams_dictionary[new_gram] += 1

            self.word_n_grams.append(joined_n_grams.rstrip())

    def produce_char_grams(self, n):
        self.char_n_grams = []
        for document in self.corpus:
            joined_n_grams = ''
            for word in document.split():
                n_grams_sentences = ngrams(list(word), n)
                for n_grams_sentence in n_grams_sentences:
                    joined_n_grams = joined_n_grams + ''.join(n_grams_sentence) + ' '

            self.char_n_grams.append(joined_n_grams.rstrip())

    def produce_stem(self):
        self.word_stem_dictionary = {}
        self.word_stem_dictionary = defaultdict(lambda:0, self.word_stem_dictionary)
        ps = PorterStemmer()
        self.stemmed = []
        for document in self.corpus:
            stemmed_document = ''
            for word in document.split():
                stem_word = ps.stem(word)
                stemmed_document += stem_word + ' '
                self.word_stem_dictionary[stem_word] += 1
            self.stemmed.append(stemmed_document.rstrip())

    def most_frequent_word_grams(self, n):
        stats = Counter(self.word_grams_dictionary)
        return stats.most_common(n)

    def most_frequent_stem_word(self, n):
        stats = Counter(self.word_stem_dictionary)
        return stats.most_common(n)
