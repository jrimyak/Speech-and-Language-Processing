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
    def __init__(self, encoder, decoder, input_embedding_layer, output_embedding_layer, input_indexer, output_indexer, max_output_len):
        # raise Exception("implement me!")
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding_layer = input_embedding_layer
        self.output_embedding_layer = output_embedding_layer
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.max_pred_len = max_output_len


    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        # getting test input data from tthe padded input tensor method 
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, True)
        # defining the SOS token as a tensor and passing that through to get the embeddeding representation of that
        SOS_index = torch.tensor(output_indexer.index_of("<SOS>")).unsqueeze(0)
        SOS_embedding = self.output_embedding_layer.forward(SOS_index).unsqueeze(0)
        EOS_index = torch.tensor(output_indexer.index_of("<EOS>")).unsqueeze(0)
        # defining the list of predictions 
        preds = []
        for index in range(len(test_data)):
            # creating a list tof output tokena
            output_tokens = []
            # creating the tensor of input tokens 
            input_tensor = torch.tensor(all_test_input_data[index]).unsqueeze(0)
            # tensor containing the length of each sentence in the batch
            input_length = torch.tensor((np.array(len(test_data[index].x_tok)))).unsqueeze(0)
            # using the encode_input_for_decoder to get the encoder outputs and the encoder final states which is the h and c tuple
            output_encoding_all, _, encode_final = encode_input_for_decoder(input_tensor, input_length, self.input_embedding_layer, self.encoder)
            # getting the output and last hidden representation
            output, hidden = self.decoder.forward(SOS_embedding, torch.tensor(1).unsqueeze(0), encode_final)
            # creating a tuple of the hidden rep
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            # getting the output tindex which is the max of the output from the decoder 
            out_index = torch.max(output, dim=1)[1]
            # loop through the decoder network until we reach EOS or max pred lengtht
            i = 0
            while out_index != EOS_index and i <= self.max_pred_len:
                i += 1
                # append the outputt token from indexer 
                output_tokens.append(self.output_indexer.get_object(out_index.item()))
                # get the embedding representation
                out_embedding_rep = self.output_embedding_layer.forward(out_index).unsqueeze(0)
                # moving forward through the decoder network and getting both the hidden rep as a tuple and the output index
                output, hidden = self.decoder.forward(out_embedding_rep, torch.tensor([1]), hidden)
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                out_index = torch.max(output, dim=1)[1]
            preds.append([Derivation(test_data[index], 1.0, output_tokens)])

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

    # Create model
    # dimensions are 200 for the embbeding layer 
    # can change the arguments to mess around with different sizes for the network 
    input_embedding_layer = EmbeddingLayer(100, len(input_indexer), 0.2)
    output_embedding_layer = EmbeddingLayer(100, len(output_indexer),0.2)
    # dimensions are 400 for the encoder and decoder and takes in the 200 dimension embedding layer
    encoder = RNNEncoder(100, 200, False)
    decoder = Decoder(100, 200, len(output_indexer))
    # getting the SOS and EOS index 
    SOS_index = torch.tensor(output_indexer.index_of("<SOS>")).unsqueeze(0)
    EOS_index = output_indexer.index_of("<EOS>")
    # init the parameters, optimizer, and criterion for the training loop
    params = list(decoder.parameters()) + list(encoder.parameters()) + list(input_embedding_layer.parameters()) + list(output_embedding_layer.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # loop over epochs 
    for epoch in range(args.epochs):
        # get the indices of the examples then shuffle that list such that we get a random ordering for our examples
        example_indices = [i for i in range(all_train_input_data.shape[0])]
        random.shuffle(example_indices)
        loss = 0
        count = 0
        for index in example_indices:
            # create a tensor that has gradient enabled to take derivatives
            loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            # zero out the grad to update the param becasue PyTorch accumulates the gradients on subsequent backward passes
            decoder.zero_grad()
            encoder.zero_grad()
            input_embedding_layer.zero_grad()
            output_embedding_layer.zero_grad()
            # creating the tensor of input tokens 
            input_tokens = torch.tensor(all_train_input_data[index]).unsqueeze(0)
            # tensor containing the length of each sentence in the batch
            input_length = torch.tensor((np.array(len(train_data[index].x_tok)))).unsqueeze(0)
            # using the encode_input_for_decoder to get the encoder outputs and the encoder final states which is the h and c tuple
            encoded_output_each_word, _, encoder_final_states = encode_input_for_decoder(input_tokens, input_length, input_embedding_layer, encoder)
            # getting the embedded form of the input words from the start of sentence index
            SOS_embedded_rep = output_embedding_layer.forward(SOS_index).unsqueeze(0)
            # getting the output and last hidden representation
            output, hidden = decoder.forward(SOS_embedded_rep, torch.tensor([1]), encoder_final_states)
            # From week 6, slide deck 2: "loop through until you reach a gold stopping point"
            # in this case, our gold stopping point is the EOS token's index, loop over all the targets find the loss
            for j in train_data[index].y_indexed:
                # updating loss by adding the loss tensor with the Cross entropy loss of the output (class scores), a tensor containing the target
                loss = loss + criterion(output, torch.tensor([j]).long())
                # checking to see if the gold tag is found, if so breaking out of the loop
                if j == EOS_index:
                    break
                # get our hidden state tuple to pass to the next layer
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                # getting the embedded form of the i-th position of the loop
                inp_embedding_rep = output_embedding_layer.forward(torch.tensor(j).unsqueeze(0)).unsqueeze(0)
                # moving to the next state in the RNN 
                output, hidden = decoder.forward(inp_embedding_rep, torch.tensor(1).unsqueeze(0), hidden)
            count+=1
            loss.backward()
            optimizer.step()
        print('Epoch: ' + str(epoch) + ' Loss: ' + str(loss/count))

    return Seq2SeqSemanticParser(encoder, decoder, input_embedding_layer, output_embedding_layer, input_indexer, output_indexer, output_max_len)



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


