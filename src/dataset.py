from collections import Counter
import numpy as np
import torch.utils.data
import json
import random

NUM_IB_LABELS = 5
WINDOW_SIZE = 6


def downsample(windows, targets, ids, ratio=5):
    # seperate_pos, nex examples

    neg = [[window, target, id] for window, target, id in
           zip(windows, targets, ids) if target == 0]

    pos = [[window, target, id] for window, target, id in
           zip(windows, targets, ids) if target == 1]

    print("init num pos: {}\nnum neg: {}".format(len(pos), len(neg)))

    random.shuffle(neg)

    neg = neg[:len(pos)*ratio]

    final_data = pos + neg

    random.shuffle(final_data)

    return [row[0] for row in final_data], \
           [row[1] for row in final_data], \
           [row[2] for row in final_data]


def get_target(example_labels, i, j):
    start_label = example_labels[i]
    end_label = example_labels[i+j-1]

    if start_label.find("(") == -1 or end_label.find(")") == -1:
        return 0

    start_mentions = set([mention.strip("(").strip(")") for mention in start_label.split("|")])
    end_mentions = set([mention.strip("(").strip(")") for mention in end_label.split("|")])

    if start_mentions & end_mentions:
        return 1
    else:
        return 0


def make_window_ex(example_sentences, example_labels):
    windows = list()
    targets = list()

    # inputs for window: (start_idx, window size)
    # for each starting token, and window size (every possible window)
    # (0) is window size 1, and it goes up to window size 6
    for i, _ in enumerate(example_sentences):
        for j in range(1, WINDOW_SIZE+1):
            if i + j > len(example_sentences):
                break
            # add the window
            windows.append((i, j))
            # add the label for that window
            targets.append(get_target(example_labels, i, j))

    return windows, targets


def make_ex(inputs, windows, targets, ids, file_sentences, file_labels, file_sent_ids, file_id):
    # for each sentence in file, make examples for it
    for sentence, label, id in zip(file_sentences, file_labels, file_sent_ids):
        ex_windows, ex_targets = make_window_ex(sentence, label)

        ex_id = "{}:{}".format(id, file_id)

        # add windows, targets, and ex_ids
        windows += ex_windows
        targets += ex_targets
        ids += [ex_id]*len(ex_windows)
        # add input sentence to mapping with ex_id
        inputs[ex_id] = sentence


# get history, respinse data from csv file
def read_file(filename, max_len, train):
    """
     for each example, create a list of words, and a list of tuples as
     labels for each word for each sentence. Create examples that a list of multiple
     sentences,3
    """

    inputs = dict()     # A dict of the sentence id in ids to sentence
    windows = list()    # list of tuples for the windows (start_idx, size)
    targets = list()    # wether or not the window is a true mention
    ids = list()        # a list of ids for sentences for each window, target

    with open(filename, 'r') as fp:
        data = json.load(fp)

    for file_id, sentences in data.items():
        file_sentences = list()
        file_labels = list()
        file_sent_ids = list()
        for sentence_id, words in sentences.items():
            # use up to max len -4 to account for cls, eos, and window tokens
            file_sentences.append([word['word'] for word in words[:max_len-4]])
            file_labels.append([word["coref"] for word in words[:max_len-4]])
            file_sent_ids.append(sentence_id)
        if not file_labels:
            continue

        # create_examples
        make_ex(inputs, windows, targets, ids, file_sentences,
                file_labels, file_sent_ids, file_id)

    return inputs, windows, targets, ids

class Vocab(object):
    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self.nb_tokens = 0

        # vocab mapping
        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

    # updates the vocab with an example
    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = self.nb_tokens
                self.id2token[self.nb_tokens] = token
                self.nb_tokens += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    # prune the vocab that occur less than the min count
    def prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set(
            [t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete -= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, self.nb_tokens))

    # load token2id from json file, useful when using pretrained model
    def load_from_dict(self, filename):
        with open(filename, 'r') as f:
            self.token2id = json.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

    # Save token2id to json file
    def save_to_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.token2id, f)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return 'Vocab: {} tokens'.format(self.nb_tokens)


class DialogueDataset(torch.utils.data.Dataset):
    START = "<START>"
    END = "<END>"
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    SEP_WORD = '<s>'
    EOS_WORD = '</s>'
    CLS_WORD = '<cls>'

    def __init__(self, filename, max_len, window_size, vocab=None,
                 update_vocab=True):
        """
        Initialize the dialogue dataset.

        Get examples, and create/update vocab

        Examples:
            History: <cls> hello ! <s> hi , how are you ? </s>
            Resoponse: <cls> i am good , thank you ! </s>

        Args:
            filename: Filename of csv file with the data
            sentence_len: Maximum token length for the history. Will be
                pruned/padded to this length
            response_len: Maximum length for the response.
            vocab: Optional vocab object to use for this dataset
            update_vocab: Set to false to not update the vocab with the new
                examples
        """
        WINDOW_SIZE = window_size

        if filename.find("train") != -1:
            self.train = True
        else:
            self.train = False

        if filename is not None:
            self.inputs, self.windows, self.targets, self.ids = read_file(
                filename, max_len, self.train)

        if self.train:
            self.windows, self.targets, self.ids = downsample(
                self.windows, self.targets, self.ids)

        self.max_len = max_len

        if vocab is None:
            # Create new vocab object
            self.vocab = Vocab(special_tokens=[DialogueDataset.START,
                                               DialogueDataset.END,
                                               DialogueDataset.PAD_WORD,
                                               DialogueDataset.UNK_WORD,
                                               DialogueDataset.SEP_WORD,
                                               DialogueDataset.EOS_WORD,
                                               DialogueDataset.CLS_WORD])
        else:
            self.vocab = vocab

        # do not want to update vocab for running old model
        if update_vocab:
            for example in self.inputs.values():
                self.vocab.add_documents(example)

    def _process_input(self, window, id):
        """
        creates token encodings for the word embeddings, positional encodings,
        and segment encodings for the dialogue history

        Examples:
            History: <cls> hello ! <s> hi , how are you ? </s>
            self.sentence_len = 15

            h_seq = np.array([4, 34, 65, 2, 23, 44, 455, 97, 56, 10, 3, 0, 0, 0, 0])
            h_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0)]
            h_seg = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0)]

        Args:
            history: list of tokens in the history
        Returns:
            h_seq: token encodings for the history
            h_pos: positional encoding for the history
            h_seg: segment encoding for the history
        """
        inputs = [DialogueDataset.CLS_WORD]
        inputs+= self.inputs[id]
        inputs = inputs[:self.max_len-2]
        inputs[-1] = DialogueDataset.EOS_WORD
        # add 1 to account for cls token that was added
        inputs.insert(window[0]+1, DialogueDataset.START)
        # add 2 to account for cls, and start
        # the window is 1:6 so it should work out
        inputs.insert(window[0]+window[1] + 2, DialogueDataset.END)

        needed_pads = self.max_len - len(inputs)
        if needed_pads > 0:
            inputs = inputs + [DialogueDataset.PAD_WORD] * needed_pads

        inputs = [
            self.vocab[token] if token in self.vocab else self.vocab[
                DialogueDataset.UNK_WORD]
            for token in inputs
        ]

        # create position embeddings, make zero if it is the pad token (0)
        h_pos = np.array([pos_i + 1 if w_i != 0 else 0
                          for pos_i, w_i in enumerate(inputs)])

        # create segment embeddings
        # 1 for not inside window, 2 for inside window, 0 for padding
        seg = list()
        i = 1
        for j, token in enumerate(inputs):
            if token == self.vocab[DialogueDataset.PAD_WORD]:
                break
            if token == self.vocab[DialogueDataset.START]:
                i += 1
            seg.append(i)
            if token == self.vocab[DialogueDataset.END]:
                i -= 1
        seg += [0] * needed_pads
        h_seg = np.array(seg, dtype=np.long)

        h_seq = np.array(inputs, dtype=np.long)

        return h_seq, h_pos, h_seg


    def __getitem__(self, index):
        """
            returns the features for an example in the dataset

        Args:
            index: index of example in dataset

        Returns:
            h_seq: token encodings for the history
            h_pos: positional encoding for the history
            h_seg: segment encoding for the history
            r_seq: token encodings for the response
            r_pos: positional encoding for the response
        """
        i_seq, i_pos, i_seg = self._process_input(self.windows[index], self.ids[index])
        label = np.array(self.targets[index], dtype=np.long)
        id = self.ids[index]
        return i_seq, i_pos, i_seg, label, id, self.windows[index]

    def __len__(self):
        return len(self.ids)
