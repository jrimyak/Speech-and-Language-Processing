# models.py

from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# getting stopwords from nltk
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

    def add_feats(self, list_of_words: List[str]):
        for word in list_of_words:
            # check to see if its not in stops words then add to indexer 
            if word.lower() not in stop_words:
                self.indexer.add_and_get_index(word.lower())

    def extract_features(self, ex_words: List[str], add_to_index: bool = False) -> List[int]:
        c = Counter() # counter to store sparse vector 
        for ex_word in ex_words:
            word = ex_word.lower()
            # adding one if the corresponding index is in the list 
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

    def add_feats(self, list_of_words: List[str]):
        for i in range(len(list_of_words)-1):
            # bigram is two adjacent wordds then add to the indexer
            word_pair = list_of_words[i].lower() + list_of_words[i+1].lower()
            self.indexer.add_and_get_index(word_pair)

    def extract_features(self, ex_words: List, add_to_index: bool = False) -> List[int]:
        c = Counter() # counter for the sparse vector 
        for i in range(len(ex_words)-1):
            word_pair = ex_words[i].lower() + ex_words[i+1].lower()
            # adding one if the corresponding index is in the list 
            if self.indexer.contains(word_pair):
                k = self.indexer.index_of(word_pair)
                c.update([k])
        return list(c.items())


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!

    Combined unigram and bigrams 
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def add_feats(self, list_of_words: List[str]):
        for i in range(len(list_of_words)-1):
            # addding both the unigram and the bigram to the indexer 
            word_pair = list_of_words[i].lower() + list_of_words[i+1].lower()
            self.indexer.add_and_get_index(word_pair)
            self.indexer.add_and_get_index(list_of_words[i].lower())
        self.indexer.add_and_get_index(list_of_words[len(list_of_words)-1].lower())

    def extract_features(self, ex_words: List, add_to_index: bool = False) -> List[int]:
        c = Counter()
        for i in range(len(ex_words)-1):
            # adding one if the corresponding index for the bigram or unigram is in the list 
            word_pair = ex_words[i].lower() + ex_words[i+1].lower()
            if self.indexer.contains(word_pair):
                k = self.indexer.index_of(word_pair)
                c.update([k])
            if self.indexer.contains(ex_words[i].lower()):
                k = self.indexer.index_of(ex_words[i].lower())
                c.update([k])
        if self.indexer.contains(ex_words[len(ex_words)-1].lower()):
            k = self.indexer.index_of(ex_words[len(ex_words)-1].lower())
            c.update([k])
        return list(c.items())

    '''
    Trigram attempt - lower percent didn't work  
    def add_feats(self, ex_words: List[str]):
        for i in range(len(ex_words)-2):
            # remove stop works 
            # if ex_words[i].lower() not in stop_words or ex_words[i+1].lower() not in stop_words:
            word_pair = ex_words[i].lower() + ex_words[i+1].lower() + ex_words[i+2].lower()
            self.indexer.add_and_get_index(word_pair)

    def extract_features(self, ex_words: List, add_to_index: bool = False) -> List[int]:
        c = Counter()
        for i in range(len(ex_words)-2):
            word_pair = ex_words[i].lower() + ex_words[i+1].lower() + ex_words[i+2].lower()
            if self.indexer.contains(word_pair):
                k = self.indexer.index_of(word_pair)
                c.update([k])
        return list(c.items())
    '''
    

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
        # joining words to make a sentence 
        sent = ''.join(ex_words)
        # creating a feature vector if the features dict doesnt have the sentence, otherwise uses that as feature vector 
        if sent not in self.features:
            f_x = self.feature_extractor.extract_features(ex_words)
            self.features[sent] = f_x
        else:
            f_x = self.features[sent]
        return f_x

    def sigmoid(self, x):
        # calculating the sigmoid function
        output = 1/(1 + np.exp(-x))
        return output

    def predict(self, ex_words: List[str]) -> int:
        # get all the features 
        f_x = self.get_feats(ex_words)
        weight_fx = 0
        # get weights multiplied by the value of features
        for k, v in f_x:
            weight_fx += self.weights[k] * v
        # find the value from the sigmoid function
        p = self.sigmoid(weight_fx)
        # if it's greater than 0.5 predict 1 otherwise predict 0
        if p > 0.5:
            return 1
        return 0

    def update(self, ex_words: List, y, y_pred, alpha):
        # get features, weighted feature vector then calc sigmoid
        f_x = self.get_feats(ex_words)
        weight_fx = 0
        for k, v in f_x:
            weight_fx += self.weights[k] * v
        prob = self.sigmoid(weight_fx)
        # gradient descent 
        '''
        loss = -[y log sigma(w . x + b) + (1-y) log(sigma (w. x + b))]
        take derivative w.r.t weight to optimize 
        d loss/dw = x_j[ sigma(w.x+b) - y]
        '''
        for key, x_j in f_x:
            self.weights[key] = self.weights[key] - \
                alpha * ( (prob - y) * x_j)

    def get_loss(self, train_exs):
        loss_sum = 0
        for ex in train_exs:
            # get features, weighted featt, the label, and calc sigmoid
            y_lab = ex.label
            f_x = self.get_feats(ex.words)
            weight_fx = 0
            for k, v in f_x:
                weight_fx += self.weights[k] * v
            prob = self.sigmoid(weight_fx)
            # calc loss and add to the sum - log at 0 error so multiplying label by 1e-5
            loss = - y_lab * np.log(prob) - (1 - y_lab*1e-5) * np.log(1 - prob*1e-5)
            loss_sum += loss
        # normalize the loss sum
        return loss_sum / float(len(train_exs)) 


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
    # use feature extractor ot get features and create the model
    for ex in train_exs:
        feat_extractor.add_feats(ex.words)

    model = LogisticRegressionClassifier(feat_extractor)
  
    epochs = 20
    alpha = 0.1
    # loop through epochs by shuffling data, getting examples, then updating the model
    for epoch in range(epochs):
        random.shuffle(train_exs)
        sample_exs = train_exs[:len(train_exs)]
        for ex in sample_exs:
            y = ex.label
            y_pred = model.predict(ex.words)
            model.update(ex.words, y, y_pred, alpha)
    return model

    # plotting average training loss and dev accuracy (used sentiment classifer for accuracy)
    # x = np.arange(0,epochs-1)
    # learning_rate = .99
    # avg_losses = np.zeros(epochs-1)
    # dev_exs = read_sentiment_examples('data/dev.txt')

    # for i in range(epochs):
    #     for ex in train_exs:
    #         y = ex.label
    #         y_pred = model.predict(ex.words)
    #         model.update(ex.words, y, y_pred, learning_rate)
    #   #  avg_losses[i-1] = model.get_loss(train_exs)
    #     avg_losses[i-1] = evaluate(model, dev_exs)
    # plt.plot(x, avg_losses,label="Step=.99")
    # model = LogisticRegressionClassifier(feat_extractor)
    # x_1 = np.arange(0,epochs-1)
    # learning_rate = .1
    # avg_losses_1 = np.zeros(epochs-1)
    # for i in range(epochs):
    #     for ex in train_exs:
    #         y = ex.label
    #         y_pred = model.predict(ex.words)
    #         model.update(ex.words, y, y_pred, learning_rate)
    # #    avg_losses_1[i-1] = model.get_loss(dev)
    #     avg_losses_1[i-1] = evaluate(model, dev_exs)
        
    # plt.plot(x_1, avg_losses_1, label="Step=0.1")
    # model = LogisticRegressionClassifier(feat_extractor)

    # x_2 = np.arange(0,epochs-1)
    # learning_rate = .02
    # avg_losses_2 = np.zeros(epochs-1)
    # for i in range(epochs):
    #     for ex in train_exs:
    #         y = ex.label
    #         y_pred = model.predict(ex.words)
    #         model.update(ex.words, y, y_pred, learning_rate)
    #  #   avg_losses_2[i-1] = model.get_loss(train_exs)
    #     avg_losses_2[i-1] = evaluate(model, dev_exs)
    # plt.plot(x_2, avg_losses_2, label="Step=0.02")
    # plt.legend(loc='upper right')
    # plt.xlabel("Batch")
    # plt.ylabel("Accuracy")
    # plt.xticks(np.linspace(0,34,35))
    # plt.show()



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
