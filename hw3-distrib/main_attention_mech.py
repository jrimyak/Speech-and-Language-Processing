import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, model_enc, model_dec, model_input_emb, model_output_emb, input_indexer, output_indexer, reverse_input, max_output_len):
        # raise Exception("implement me!")
        self.encoder = model_enc
        self.decoder = model_dec
        self.input_embeddings = model_input_emb
        self.output_embeddings = model_output_emb
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.reverse_input = reverse_input
        self.max_pred_len = max_output_len


    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        self.encoder.eval()
        self.decoder.eval()
        self.input_embeddings.eval()
        self.output_embeddings.eval()
        input_max_len = np.max(np.asarray(
            [len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(
            test_data, self.input_indexer, input_max_len, self.reverse_input)
        SOS_idx = torch.tensor(
            output_indexer.index_of("<SOS>")).unsqueeze(0)
        SOS_embed = self.output_embeddings.forward(SOS_idx).unsqueeze(0)
        EOS_idx = torch.tensor(
            output_indexer.index_of("<EOS>")).unsqueeze(0)
        num_examples = len(test_data)
        preds = []
        print("In decode")
        for idx in range(num_examples):

            out_toks = []
            inp_ex = torch.tensor(all_test_input_data[idx]).unsqueeze(0)
            inp_len = torch.tensor(
                (np.array(len(test_data[idx].x_tok)))).unsqueeze(0)
            enc_output_each_word, enc_context_mask, enc_final_states = encode_input_for_decoder(
                inp_ex, inp_len, self.input_embeddings, self.encoder)
            class_scores, dec_hid = self.decoder.forward(SOS_embed, torch.tensor(
                1).unsqueeze(0), enc_final_states, enc_output_each_word)
            dec_hid = (dec_hid[0].unsqueeze(0), dec_hid[1].unsqueeze(0))
            out_idx = torch.max(class_scores, dim=1)[1]
            i = 0
            while out_idx != EOS_idx and i <= self.max_pred_len:
                i += 1
                out_toks.append(self.output_indexer.get_object(out_idx.item()))
                out_emb = self.output_embeddings.forward(out_idx).unsqueeze(0)
                class_scores, dec_hid = self.decoder.forward(
                    out_emb, torch.tensor(1).unsqueeze(0), dec_hid, enc_output_each_word)
                dec_hid = (dec_hid[0].unsqueeze(0), dec_hid[1].unsqueeze(0))
                out_idx = torch.max(class_scores, dim=1)[1]
            preds.append([Derivation(test_data[idx], 1.0, out_toks)])

        return preds
        

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb: EmbeddingLayer, model_enc: RNNEncoder):
    """
    Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
    inp_lens_tensor lengths.
    YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
    as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
    :param x_tensor: [batch size, sent len] tensor of input token indices
    :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
    :param model_input_emb: EmbeddingLayer
    :param model_enc: RNNEncoder
    :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
    are real and which ones are pad tokens), and the encoder final states (h and c tuple)
    E.g., calling this with x_tensor (0 is pad token):
    [[12, 25, 0, 0],
    [1, 2, 3, 0],
    [2, 0, 0, 0]]
    inp_lens = [2, 3, 1]
    will return outputs with the following shape:
    enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
    enc_final_states = 3 x dim
    """
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=True)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words, call
    # the encoder, call your decoder, accumulate losses, update parameters
    # raise Exception("Implement the rest of me to train your encoder-decoder model")


    # Create model
    model_input_emb = EmbeddingLayer(
        200, len(input_indexer), 0.2)
    model_enc = RNNEncoder(200, 400,\
                          True)
    model_output_emb = EmbeddingLayer(
        200, len(output_indexer),0.2)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    # model_dec = DecoderCell(args.input_dim, args.hidden_size, len(output_indexer), dropout=0)
    model_attn_dec = AttentionDecoderCell(
        200, 400, len(output_indexer), input_max_len)

    SOS_idx = torch.tensor(output_indexer.index_of("<SOS>")).unsqueeze(0)
    EOS_idx = output_indexer.index_of("<EOS>")
    num_examples = all_train_input_data.shape[0]


    #params = list(model_dec.parameters()) + list(model_enc.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    params = list(model_attn_dec.parameters()) + list(model_enc.parameters()) + \
        list(model_input_emb.parameters()) + \
        list(model_output_emb.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        print("epoch:", epoch)

        ex_indices = [i for i in range(num_examples)]
        random.shuffle(ex_indices)
        loss = 0
        count = 0
        for idx in ex_indices:
            loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            model_attn_dec.zero_grad()
            model_enc.zero_grad()
            model_input_emb.zero_grad()
            model_output_emb.zero_grad()

            inp_ex = torch.tensor(all_train_input_data[idx]).unsqueeze(0)
            inp_len = torch.tensor(
                (np.array(len(train_data[idx].x_tok)))).unsqueeze(0)
            gold_len = len(train_data[idx].y_indexed)
            enc_output_each_word, enc_context_mask, enc_final_states = encode_input_for_decoder(
                inp_ex, inp_len, model_input_emb, model_enc)

            SOS_embed = model_output_emb.forward(SOS_idx).unsqueeze(0)
            class_scores, dec_hid = model_attn_dec.forward(SOS_embed, torch.tensor(
                1).unsqueeze(0), enc_final_states, enc_output_each_word)

            for gold_idx in train_data[idx].y_indexed:
                loss = loss + criterion(class_scores, torch.LongTensor([gold_idx]))

                if gold_idx == EOS_idx:
                    break
                dec_hid = (dec_hid[0].unsqueeze(0), dec_hid[1].unsqueeze(0))
                prev_idx = torch.max(class_scores, dim=1)[1]
                inp_emb = model_output_emb.forward(
                    torch.tensor(gold_idx).unsqueeze(0)).unsqueeze(0)
                class_scores, dec_hid = model_attn_dec.forward(
                    inp_emb, torch.tensor(1).unsqueeze(0), dec_hid, enc_output_each_word)
            count+=1
            loss.backward()
            optimizer.step()
        print('loss ' + str(loss/count))

    return Seq2SeqSemanticParser(model_enc, model_attn_dec, model_input_emb, model_output_emb, input_indexer, output_indexer, True, output_max_len)



def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    java_crashes = False
    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        evaluate(dev_data_indexed, decoder)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")


