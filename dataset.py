import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, field
from collections import namedtuple
from tqdm import tqdm

tqdm.pandas()

default_split_kwargs = {'train_size': 0.8,
                        'dev_size': 0.1,
                        'test_size': 0.1,
                        'random_state': 1}
default_vec_kwargs = {'max_features': 2000,
                      'stop_words':'english'}
default_clf_kwargs = {}  


field_kwargs = {'default':None, 'repr':False, 'compare':False}
split_tuple = namedtuple('splits', ['train', 'dev', 'test'])
vec_tuple = namedtuple('vectors', ['train', 'dev', 'test'])
context_attribute_pair = namedtuple('ca_pair', ['context', 'attributes'])

strip_nonalphanumeric = re.compile('([^\s\w]|_)+')


class Dataset:

    def __init__(self, name, path='', text_col='response_text',
                 label_col='responder_gender', split_kwargs=None,
                 vec_kwargs=None, clf_kwargs=None):
        # Check for default params
        if split_kwargs is None:
            split_kwargs = default_split_kwargs
        if vec_kwargs is None:
            vec_kwargs = default_vec_kwargs
        if clf_kwargs is None:
            clf_kwargs = default_clf_kwargs

        self.name = name
        self.label_col = label_col

        print("Loading the dataset...")
        self.df = self.load(name, path)

        print("Cleaning data...")
        self.df['cleaned_text'] = self.df[text_col].apply(self._clean_text)
        self.text_col = 'cleaned_text'

        print("Making data splits...")
        self.splits = self.make_splits(self.df, **split_kwargs)

        print("Training the vectorizer...")
        self.vectorizer = self.make_vectorizer(vec_kwargs)

        print("Transforming the vectors...")
        self.vecs = self.vectorize()
        self.labels = self.vectorize_labels()

        print("Training a base classifier...")
        self.clf = self.make_classifier(clf_kwargs)

        print(f"Done initializing {self.name} dataset!")


    def load(self, dataset, path='data/rt_gender/'):
        """Load a specific dataset."""

        if dataset == "enron":
            return self._load_enron(path)
        else:
            return self._load_rtgender(dataset, path)

    def _load_enron(self, path):
        """Function for loading the enron dataset."""
        raise NotImplementedError

    def _load_rtgender(self, dataset, path):
        """Function for loading rtgender datasets."""
        fname = f"{path}{dataset}.csv"
        try:
            return pd.read_csv(fname, index_col='post_id')
        except FileNotFoundError:
            raise ValueError(f"Dataset {fname} does not exist.")

    def _clean_text(self, text):
        return strip_nonalphanumeric.sub('', text.lower())

    def make_splits(self, df, train_size, dev_size, test_size,
                    random_state):
        """Make train/dev/test splits of a dataframe."""
        if train_size + dev_size + test_size != 1.0:
            raise ValueError('Splits must add to 100\%')
        
        # Split out the test set
        df_train, df_test = train_test_split(df, test_size=test_size,
            train_size=1-test_size, random_state=random_state)
        
        # Re-scale the sizes to split the train/dev appropriately
        new_train_size = train_size / (1 - test_size)
        new_dev_size = 1 - new_train_size

        df_train, df_dev = train_test_split(df_train, test_size=new_dev_size,
            train_size=new_train_size, random_state=random_state)
        
        return split_tuple(df_train, df_dev, df_test)

    def make_vectorizer(self, vec_kwargs):
        """Make a Tf-idf vectorizer on the given corpus."""
        vectorizer = TfidfVectorizer(**vec_kwargs)
        vectorizer.fit(self.splits.train[self.text_col])
        return vectorizer

    def vectorize(self):
        """Vectorize the various splits of a dataset."""
        train_vec = self.vectorizer.transform(self.splits.train[self.text_col])
        dev_vec = self.vectorizer.transform(self.splits.dev[self.text_col])
        test_vec = self.vectorizer.transform(self.splits.test[self.text_col])

        return vec_tuple(train_vec, dev_vec, test_vec)

    def vectorize_labels(self):
        """Vectorize the labels in the dataset."""
        mapping = {'M': 0, 'W': 1}
        to_vec = lambda col: np.array([mapping[g] for g in col])

        train_labels = to_vec(self.splits.train[self.label_col])
        self.splits.train['label_col'] = train_labels
        dev_labels = to_vec(self.splits.dev[self.label_col])
        self.splits.dev['label_col'] = dev_labels
        test_labels = to_vec(self.splits.test[self.label_col])
        self.splits.test['label_col'] = test_labels

        self.df['label_col'] = to_vec(self.df[self.label_col])

        return vec_tuple(train_labels, dev_labels, test_labels)

    def make_classifier(self, clf_kwargs):
        """Fit a basic Log Reg classifier to the data."""
        return LogisticRegression(**clf_kwargs).fit(
            self.vecs.train, self.labels.train)

    def __len__(self):
        return sum([len(s) for s in self.splits])


class TransformedDataset:
    
    def __init__(self, dataset, gamma, lambd=1, pre_comp_word_counts=None):
        
        self.gamma = gamma
        self.lambd = lambd
        self.all_labels = set(dataset.df['label_col'])
        
        # Determine word-counts
        if pre_comp_word_counts is None:
            print("Computing word counts...")
            self.wc_df = self.make_word_counts_df(dataset.splits.train)
        else:
            print("Loading word counts...")
            self.wc_df = pre_comp_word_counts
        
        # Make context-attribute pairs
        print("Transforming Splits")
        self.male_splits = split_tuple(*[self.split_ca(split, 0)
                                         for split in dataset.splits])
        self.female_splits = split_tuple(*[self.split_ca(split, 1)
                                         for split in dataset.splits])

    def make_word_counts_df(self, df):
        """Get out the naive-bayes counts for all words."""
        
        # Split by_label
        dfs = {label: df[df['label_col'] == label]
                     for label in self.all_labels}

        # Count words for each
        vectorizers = {label: CountVectorizer(min_df=10).fit(
                                dfs[label]['response_text'])
                       for label in self.all_labels}

        # Convert to DataFrame
        word_counts = {label: np.array(vectorizers[label].transform(
                                [' '.join(dfs[label]['response_text'])]).todense())[0]
                       for label in self.all_labels}
        features = {label: vectorizers[label].get_feature_names()
                    for label in self.all_labels}
        count_dfs = {label: pd.DataFrame(word_counts[label],
                                         index=features[label],
                                         columns=[f"{label}_count"])
                     for label in self.all_labels}
        wc_df = pd.concat([count_df for count_df in count_dfs.values()], axis=1)
        wc_df.fillna(0, inplace=True)
        
        # Determine the per-label ratio for each word
        all_count_cols = [f"{label}_count" for label in self.all_labels]
        for label in self.all_labels:
            wc_df[f"{label}_ratio"] = np.log(
                (wc_df[f"{label}_count"] + self.lambd) /
                (wc_df[all_count_cols].sum(axis=1) - wc_df[f"{label}_count"] + self.lambd))

        return wc_df
    
    def split_ca(self, df, label):
        """Split each sentence into its context and its attributes."""            
        df = df[df['label_col'] == label]
        ca_pairs = df['cleaned_text'].progress_apply(self._split_ca, args=(label,))
            
        return ca_pairs
    
    def _split_ca(self, text, label):
        words = text.split(' ')
        scores = [self.score_word(w, label) for w in words]
        
        context = []
        attributes = []
        
        for w, score in zip(words, scores):
            if score > self.gamma:
                attributes.append(w)
            else:
                context.append(w)
                
        return context_attribute_pair(context, attributes)
    
    def score_word(self, word, label):
        try:
            return self.wc_df[f"{label}_ratio"][word]
        except KeyError:
            # Novel word
            return 0
        
    
    def save(self, fname):
        raise NotImplementedError