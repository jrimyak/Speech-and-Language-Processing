# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List
import numpy as np

class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        # Adapted from psuedocode in Figure A.9 in JM
        # create a scorer 
        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        # create matrices for viterbi and backpointers
        viterbi = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        backpointers = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        # initialize the 0th column to be the init score and the emissions score 
        for i in range(len(self.tag_indexer)):
            viterbi[0][i] = scorer.score_init(sentence_tokens, i) + scorer.score_emission(sentence_tokens, i, 0)
        # recursive step, loop through the time and steps
        for time in range(1, len(sentence_tokens)):
            for state in range(len(self.tag_indexer)):
                # create a temp matrix to find the max and argmax to update viterbi and backpointers using numpy
                temp = np.zeros(len(self.tag_indexer))
                for i in range(len(self.tag_indexer)):
                    temp[i] = viterbi[time-1][i] + scorer.score_transition(sentence_tokens,i,state) + scorer.score_emission(sentence_tokens,state,time)
                viterbi[time, state] = np.max(temp)
                backpointers[time, state] = np.argmax(temp)
        # find the argmax of the last column of the viterbi matrix then go backwards to get the full path
        pred_tags = []
        tag_index = np.argmax(viterbi[-1, :])
        pred_tags.append((self.tag_indexer.get_object(np.argmax(viterbi[-1, :]))))
        for t in range(len(sentence_tokens) - 1, 0, -1):
            pred_tags.append(self.tag_indexer.get_object(np.int(backpointers[t, tag_index])))
            tag_index = np.int(backpointers[t, tag_index])
        # reverse the list and that is the labeled sentence 
        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class FeatureBasedSequenceScorer(object):
    def __init__(self, feature_weights, feature_cache):
        self.feature_weights = feature_weights
        self.feature_cache = feature_cache

    '''
    function for scoring based on the emissions for the CRF. Can use the score_indexed_features method to find said score 
    '''
    def indexed_feat_score(self, word_index, tag_index):
        feats = self.feature_cache[word_index][tag_index]
        return score_indexed_features(feats, self.feature_weights) 
        
class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens):
        predict_tags = []
        # viterbi algorithm for inference and create score based on scorer
        # feature cache
        feature_cache = [[[] for k in range(len(self.tag_indexer))] for j in range(len(sentence_tokens))]
        for word_idx in range(len(sentence_tokens)):
            for t in range(len(self.tag_indexer)):
                feature_cache[word_idx][t] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(t), self.feature_indexer, False)
        N = len(sentence_tokens)  
        T = len(self.tag_indexer)  
        # define a scorer 
        scorer = FeatureBasedSequenceScorer(self.feature_weights, feature_cache)
        # create matrices 
        viterbi = np.zeros((N, T))
        backtrace = np.zeros((N, T))
        # create a matrix to hold all the scores from our scorer and populate it using the indexed_feat_score method 
        scores = np.zeros((N, T))
        for t in range(N):
            for j in range(T):
                scores[t, j] = scorer.indexed_feat_score(t, j)
        # initialize the 0-th column of the viterbi matrix 
        for i in range(T):
            viterbi[0, i] = scorer.indexed_feat_score(0, i)
        # loop through all time and step
        for t in range(1, N):
            for j in range(T):
                # get score, and populate np matrix to get argmax and max 
                emission_val = scores[t, j]
                temp = np.zeros(T)
                for i in range(T):
                    inval = 0
                    # get the prev and current tag to check for conditions 
                    current_tag = self.tag_indexer.get_object(j)
                    previous_tag = self.tag_indexer.get_object(i)
                    # the constraint: cannot transition to I-X from anything but I-X and B-X or O should not follow I-X and I-A/B-A should not follow I-B
                    # these conditions are implemented here 
                    if previous_tag == 'O' and current_tag[0] == 'I':
                        inval = -np.inf
                    elif current_tag[0] == 'I':
                        if previous_tag[2:] != current_tag[2:]:
                            inval = -np.inf
                    # updating the index of temp
                    temp[i] = viterbi[t-1,i] + inval + emission_val
                # finding the argmax and max and updating respectively 
                backtrace[t, j] = np.argmax(temp)
                viterbi[t, j] = np.max(temp)
        # finding the prediction tags from the argmax of the viterbi matrix and getting objects from the backtrace matrx in the loop
        tag_index = np.argmax(viterbi[-1, :])
        predict_tags.append(self.tag_indexer.get_object(tag_index))
        for i in range(N-1, 0, -1):
            predict_tags.append(self.tag_indexer.get_object(np.int(backtrace[i, tag_index])))
            tag_index = np.int(backtrace[i, tag_index])
        # reversed list is our prediction 
        predict_tags.reverse()
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(predict_tags))


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    print("Training")
    # largely adapted from slides 
    # defining weights and the UnregularizedAdagradTrainer 
    feature_weights = np.zeros(len(feature_indexer))
    optimizer = UnregularizedAdagradTrainer(feature_weights)
    for epoch in range(5):
        for sentence_index in range(len(sentences)):
            gradients = Counter()
            N = len(sentences[sentence_index])  
            T = len(tag_indexer)  
            # matrix of potential phis
            potential_phis = np.zeros((T, N))
            for y in range(T):
                for x in range(N):
                    # finding phie (yi, i, x) by taking the weights and the values in the feature cache and summing over it 
                    potential_phis[y, x] = np.sum(np.take(feature_weights, np.asarray(feature_cache[sentence_index][x][y])))
            # creating matrices to store the forward and backward probabilities 
            forward = np.zeros((T, N))  # alpha in slides 
            backward = np.zeros((T, N))  # beta in slides 
            # initialize the forward matrix with potenital phis in all rows of the 0th column 
            forward[:, 0] = potential_phis[:, 0]
            # doing the recursion step for forward 
            for x in range(1, N):
                for y in range(T):
                    # add the potential phis and the logaddexp in all rows in the x-1 column, reduce will reduce the row and take the logaddexp
                    forward[y, x] = potential_phis[y, x] + np.logaddexp.reduce(forward[:, x - 1])
            # doing the recusion step for backwards 
            for x in range(1, N):
                for y in range(T):
                    # add the potential phis and the logaddexp in all rows in the N-x column, reduce will reduce the row and take the logaddexp 
                    backward[y, N - x - 1] = np.logaddexp.reduce(backward[:, N - x] + potential_phis[:, N - x])

            # finding the normalizing Z constant, we can use the last column of the forward matrix to avoid using the last column of the backwards matrix since it contains 0
            # also since we are in logspace, reduce will reduce the row and take the logaddexp for this value 
            Z = np.logaddexp.reduce(forward[:, -1])
            # initialze matrix for posterior probs 
            posterior = np.zeros((T, N))
 
            # find the stochastic gradient for the sentence 
            for word_index in range(N):
                # find the posterior by exponentiating the sum of forward, backward, and subtracting Z at the word_index column 
                # adding and subtracting bc logspace
                # in this loop for so to avoid another loop 
                posterior[:, word_index] = np.exp(forward[:, word_index] + backward[:, word_index] - Z)
                # gradients is the sum of gold features - expected features
                # find the gold tag
                gold_tag = tag_indexer.index_of(sentences[sentence_index].get_bio_tags()[word_index])
                for feature in feature_cache[sentence_index][word_index][gold_tag]:
                    # update counter of features with gold tag 
                    gradients[feature] += 1  
                # subtract posterior from feature to represent the expected feature component of the gradient for each feature
                for tag_index in range(T):
                    for feature in feature_cache[sentence_index][word_index][tag_index]:
                        gradients[feature] -= posterior[tag_index, word_index]
            # update the weights with the optimizer provided using the gradients 
            optimizer.apply_gradient_update(gradients, 1)
    return CrfNerModel(tag_indexer, feature_indexer, optimizer.get_final_weights())

def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)

