# models.py

from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool = False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def add_feats(self, word_list: List[str]):
        # remove stop words
        for word in word_list:
            word.lower()
            # check to see if its in stops words
            if word not in stop_words:
                self.indexer.add_and_get_index(word)

    def extract_features(self, ex_words: List[str], add_to_index: bool = False) -> List[int]:
        c = Counter()
        for word in ex_words:
            word.lower()
            if self.indexer.contains(word):
                k = self.indexer.index_of(word)
                c.update([k])
        return list(c.items())


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def add_feats(self, ex_words: List[str]):
        for i in range(len(ex_words)-1):
            # remove stop works 
            word_pair = ex_words[i].lower() + ex_words[i+1].lower()
            self.indexer.add_and_get_index(word_pair)

    def extract_features(self, ex_words: List) -> List[int]:
        c = Counter()
        for i in range(len(ex_words)-1):
            word_pair = ex_words[i].lower() + ex_words[i+1].lower()
            if self.indexer.contains(word_pair):
                k = self.indexer.index_of(word_pair)
                c.update([k])
        return list(c.items())


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feature_extractor):
        SentimentClassifier.__init__(self)
        self.feature_extractor = feature_extractor
        self.indexer = self.feature_extractor.get_indexer()
        self.weights = np.zeros((self.indexer.__len__(),))
        self.features = {}

    def get_feats(self, ex_words: List[str]) -> List[int]:
        sent = ''.join(ex_words)
        if sent not in self.features:
            f_x = self.feature_extractor.extract_features(ex_words)
            self.features[sent] = f_x
        else:
            f_x = self.features[sent]
        return f_x

    def sigmoid(self, x):
        output = 1/(1 + np.exp(-x))
        # output = max(output, 1e-8)
        # output = min(output, 1-1e-8)
        return output

    def predict(self, ex_words: List[str]) -> int:
        f_x = self.get_feats(ex_words)
        wfx = 0

        for k, v in f_x:
            wfx += self.weights[k] * v

        p = self.sigmoid(wfx)
        if p > 0.5:
            return 1
        return 0

    def update(self, ex_words: List, y, y_pred, alpha):
        f_x = self.get_feats(ex_words)
        wfx = 0
        for k, v in f_x:
            wfx += self.weights[k] * v
        prob = self.sigmoid(wfx)
        '''
        update rules: write about tomoorow 
        p = log(ex/(1+ex))
        dp/dx = 1 - ex/(1+ex) = 1/(1+ex)
        loss = -y * log(p) - (1-y) * log(1-p)
        d(loss)/dw = -y * fx *(1-p) + (1-y) * fx*p
        '''
        for key, x_j in f_x:
            self.weights[key] = self.weights[key] - \
                alpha * ( (prob - y) * x_j)

    def get_loss(self, train_exs):
        loss_sum = 0
        for ex in train_exs:
            y = ex.label
            x = ex.words
            f_x = self.get_feats(x)

            wfx = 0

            for k, v in f_x:
                wfx += self.weights[k] * v

            p = self.sigmoid(wfx)
            loss = - y * np.log(p) - (1 - y) * np.log(1 - p)
            loss_sum += loss

        return loss_sum / float(len(train_exs))  # normalize


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    def __init__(self):
        raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for ex in train_exs:
        feat_extractor.add_feats(ex.words)

    model = LogisticRegressionClassifier(feat_extractor)
    epochs = 30
    alpha = 0.5
    for t in range(epochs):
        random.shuffle(train_exs)
        # sample_size = len(train_exs)
        sampled_exs = train_exs[:len(train_exs)]

        for ex in sampled_exs:
            y = ex.label
            y_pred = model.predict(ex.words)
            model.update(ex.words, y, y_pred, alpha)
    return model
  #  raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception(
            "Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
