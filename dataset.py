import pandas as pd
import numpy as np
import pickle
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

@dataclass
class Splits:
    train: list
    dev: list
    test: list

@dataclass
class ContextAttributePair:
    context: list
    attribute: list


# split_tuple = namedtuple('splits', ['train', 'dev', 'test'])
# vec_tuple = namedtuple('vectors', ['train', 'dev', 'test'])
# context_attribute_pair = namedtuple('ca_pair', ['context', 'attributes'])

strip_nonalphanumeric = re.compile('([^\s\w]|_)+')


class Dataset:

    def __init__(self, name, path='', text_col='response_text',
                 sender_col='responder_gender', recipient_col='op_gender',
                 split_kwargs=None, vec_kwargs=None, clf_kwargs=None,
                 predict_on_sender=True, sender_gender='M'):
        # Check for default params
        if split_kwargs is None:
            split_kwargs = default_split_kwargs
        if vec_kwargs is None:
            vec_kwargs = default_vec_kwargs
        if clf_kwargs is None:
            clf_kwargs = default_clf_kwargs

        self.name = name
        self.sender_col = sender_col
        self.recipient_col = recipient_col

        print("Loading the dataset...")
        self.df = self.load(name, path)
        self.vectorize_df_labels(predict_on_sender, sender_gender)

        print("Cleaning data...")
        self.df['cleaned_text'] = self.df[text_col].apply(self._clean_text)
        self.orig_text_col = text_col
        self.text_col = 'cleaned_text'

        print("Making data splits...")
        self.make_splits(**split_kwargs)

        print("Training the vectorizer...")
        self.vectorizer = self.make_vectorizer(vec_kwargs)
        self.labels = self.vectorize_labels()

        print("Transforming the vectors...")
        self.vecs = self.vectorize()

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
        df = Dataset.load_pickle(f"{path}full_data_small.pkl")

        # Mask out group communications
        msk = [(len(g1) + len(g2)) == 1 for g1, g2 in
               zip(df['to_gender'], df['cc_gender'])]
        df = df[msk]

        to_gender = [g1 + g2 for g1, g2 in
                     zip(df['to_gender'], df['cc_gender'])]
        df['to_gender'] = [g[0] for g in to_gender]

        # Mask out uncertain gender individuals
        msk = np.array([g1 != 'I' and g2 != 'I' for g1, g2 in
                                zip(df['from_gender'], df['to_gender'])])
        df = df[msk]

        return df

    def _load_rtgender(self, dataset, path):
        """Function for loading rtgender datasets."""
        fname = f"{path}{dataset}.csv"
        try:
            df = pd.read_csv(fname, index_col='post_id')
        except FileNotFoundError:
            raise ValueError(f"Dataset {fname} does not exist.")

        # Convert 'W's -> 'F's in the RTGender data
        df[self.sender_col] = [v if v == 'M' else 'F'
                               for v in df[self.sender_col]]
        df[self.recipient_col] = [v if v == 'M' else 'F'
                                  for v in df[self.recipient_col]]

        return df

    def _clean_text(self, text):
        return strip_nonalphanumeric.sub('', text.lower().strip())

    def make_splits(self, train_size, dev_size, test_size,
                    random_state):
        """Make train/dev/test splits of a dataframe."""
        if train_size + dev_size + test_size != 1.0:
            raise ValueError('Splits must add to 100\%')

        self.df['split'] = 'train'

        not_train_df = self.df.sample(frac=dev_size + test_size)

        dev_frac = dev_size / (dev_size + test_size)
        dev_idx = not_train_df.sample(frac=dev_frac).index.values

        test_idx = np.array([v for v in not_train_df.index
                             if v not in dev_idx])

        self.df.loc[dev_idx, 'split'] = 'dev'
        self.df.loc[test_idx, 'split'] = 'test'

        self._train_msk = self.df['split'] == 'train'
        self._dev_msk = self.df['split'] == 'dev'
        self._test_msk = self.df['split'] == 'test'

    @property
    def train(self):
        return self.df[self._train_msk]

    @property
    def dev(self):
        return self.df[self._dev_msk]

    @property
    def test(self):
        return self.df[self._test_msk]

    @property
    def splits(self):
        return [self.train, self.dev, self.test]
    
    def make_vectorizer(self, vec_kwargs):
        """Make a Tf-idf vectorizer on the given corpus."""
        vectorizer = TfidfVectorizer(**vec_kwargs)
        vectorizer.fit(self.train[self.text_col])
        return vectorizer

    def vectorize(self):
        """Vectorize the various splits of a dataset."""
        train_vec = self.vectorizer.transform(self.train[self.text_col])
        dev_vec = self.vectorizer.transform(self.dev[self.text_col])
        test_vec = self.vectorizer.transform(self.test[self.text_col])

        return Splits(train_vec, dev_vec, test_vec)

    def vectorize_labels(self):
        """Vectorize the labels in the dataset."""
        mapping = {'M': 0, 'F': 1}
        to_vec = lambda col: np.array([mapping[g] for g in col])

        train_labels = to_vec(self.train[self.label_col])
        dev_labels = to_vec(self.dev[self.label_col])
        test_labels = to_vec(self.test[self.label_col])

        return Splits(train_labels, dev_labels, test_labels)

    def vectorize_df_labels(self, predict_on_sender, sender_gender):
        """Vectorize the labels in the dataset."""
        mapping = {'M': 0, 'F': 1}
        to_vec = lambda col: np.array([mapping[g] for g in col])

        self.df['sender_label'] = to_vec(self.df[self.sender_col])
        self.df['recipient_label'] = to_vec(self.df[self.recipient_col])

        # Filter out depending on what prediction task we are doing
        if predict_on_sender:
            self.label_col = self.sender_col
        else:
            self.label_col = self.recipient_col
            # Filter out data if we are predicting M->F etc.
            msk = self.df[self.sender_col] == sender_gender
            self.df = self.df[msk]

    def make_classifier(self, clf_kwargs):
        """Fit a basic Log Reg classifier to the data."""
        return LogisticRegression(**clf_kwargs).fit(
            self.vecs.train, self.labels.train)

    def report_clf_results(self, show_test=False):
        """Show the classification reports from each split."""
        train_pred = self.clf.predict(self.vecs.train)
        dev_pred = self.clf.predict(self.vecs.dev)

        print("~~~ Train Split Results ~~~\n")
        self._report_clf_results(self.labels.train, train_pred)

        print("~~~ Dev Split Results ~~~\n")
        self._report_clf_results(self.labels.dev, dev_pred)

        if show_test:
            test_pred = self.clf.predict(self.vecs.test)
            print("~~~ Test Split Results ~~~\n")
            self._report_clf_results(self.labels.test, test_pred)

    def _report_clf_results(self, y_true, y_pred):
        """Show the classification results for a given split."""
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()
        print(f"Acc: {accuracy_score(y_pred, y_true):0.3f}")
        print(f"Acc: {class_acc}")
        print(classification_report(y_true, y_pred, target_names=['M', 'F']))

    def __len__(self):
        return len(self.df)

    def save_pickle(self, PATH="data/final/", light=True):
        """Save the dataset as a pickle."""
        if light:
            # Drop out original text column before saving
            self.df.drop(self.orig_text_col, axis=1, inplace=True)

        if not os.path.exists(PATH):
            os.makedirs(PATH)

        with open(f"{PATH}{self.name}_dataset.pkl", 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(fname):
        """Load a dataset from a pickle."""
        with open(fname, 'rb') as f:
            return pickle.load(f)


class TransformedDataset:
    
    def __init__(self, dataset, gamma, lambd=1, pre_comp_word_counts=None):
        
        self.gamma = gamma
        self.lambd = lambd
        self.all_labels = set(dataset.df['sender_label'])
        
        # Determine word-counts
        if pre_comp_word_counts is None:
            print("Computing word counts...")
            self.wc_df = self.make_word_counts_df(dataset.train)
        else:
            print("Loading word counts...")
            self.wc_df = pre_comp_word_counts
        
        # Make context-attribute pairs
        print("Transforming Splits")
        self.male_splits = Splits(*[self.split_ca(split, 0)
                                         for split in dataset.splits])
        self.female_splits = Splits(*[self.split_ca(split, 1)
                                         for split in dataset.splits])

    def make_word_counts_df(self, df):
        """Get out the naive-bayes counts for all words."""
        
        # Split by_label
        dfs = {label: df[df['sender_label'] == label]
                     for label in self.all_labels}

        # Count words for each
        vectorizers = {label: CountVectorizer(min_df=10).fit(
                                dfs[label]['cleaned_text'])
                       for label in self.all_labels}

        # Convert to DataFrame
        word_counts = {label: np.array(vectorizers[label].transform(
                                [' '.join(dfs[label]['cleaned_text'])]).todense())[0]
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
        df = df[df['sender_label'] == label]
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
                
        return ContextAttributePair(context, attributes)
    
    def score_word(self, word, label):
        try:
            return self.wc_df[f"{label}_ratio"][word]
        except KeyError:
            # Novel word
            return 0

    def save_pickle(self, PATH="data/final/", light=True):
        """Save the dataset as a pickle."""
        if light:
            # Drop out original text column before saving
            self.df.drop(self.orig_text_col, axis=1, inplace=True)

        if not os.path.exists(PATH):
            os.makedirs(PATH)

        with open(f"{PATH}{self.name}_dataset.pkl", 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(fname):
        """Load a dataset from a pickle."""
        with open(fname, 'rb') as f:
            return pickle.load(f)