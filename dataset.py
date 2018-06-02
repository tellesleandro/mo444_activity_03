import os
import pandas

from pdb import set_trace as bp

class Dataset:

    def __init__(self, filename):
        self.filename = filename

    def load(self):
        self.dataset = pandas.read_csv(self.filename)

    def remove_duplicates(self):
        self.dataset = self.dataset.drop_duplicates(subset = 'headline_text')
        filename = os.path.basename(self.filename)
        base, ext = os.path.splitext(filename)
        tmp_filename = base + '_non_duplicated' + ext
        self.dataset.to_csv( \
                            path_or_buf = tmp_filename, \
                            header = self.dataset.columns.tolist(), \
                            index = False
                            )
        self.dataset = pandas.read_csv(tmp_filename)

    def sample_count(self):
        return len(self.dataset)

    def headline_texts(self):
        return self.dataset.loc[:, 'headline_text']

    def split_years(self):
        self.grouped_dataset = self.dataset.groupby( \
                                [self.dataset.publish_date.astype(str).str[0:4]] \
                                )

    def save_splitted(self):
        filename = os.path.basename(self.filename)
        base, ext = os.path.splitext(filename)
        for year in self.grouped_dataset.groups.keys():
            indices = self.grouped_dataset.groups[year]
            self.dataset.loc[indices,:].to_csv( \
                            path_or_buf = base + '_' + year + ext, \
                            header = self.dataset.columns.tolist(), \
                            index = False
                            )

    def save_word_cloud(self):
        filename = os.path.basename(self.filename)
        base, ext = os.path.splitext(filename)
        file = open(base + '_word_cloud.txt', 'w')
        for headline in self.headline_texts():
            for word in headline.split():
                file.write(word + '\n')
        file.close()
