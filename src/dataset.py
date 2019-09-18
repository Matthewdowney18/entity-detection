from collections import Counter
import numpy as np
import torch.utils.data
import json


def make_train_ex(inputs, targets, ids, file_sentences, file_labels, file_sent_ids, max_len, file_id):
    example_input = list()
    example_target = list()
    example_id = list()
    for sentence, label, id in zip(file_sentences, file_labels, file_sent_ids):
        example_input.append(sentence)
        example_target.append(label)
        example_id.append(id)
        while True:
            length = sum([len(i) + 1 for i in example_input]) + 2
            # next one on
            if length > max_len:
                # delete first sentence
                example_input = example_input[1:]
                example_target = example_target[1:]
                example_id = example_id[1:]
            else:
                break
            # append last node
        inputs.append(example_input)
        targets.append(example_target)
        ids.append((example_id, file_id))


def make_val_ex(inputs, targets, ids, file_sentences, file_labels, file_sent_ids, max_len, file_id):
    if len(file_sentences[0]) + 2 < max_len:
        inputs.append(file_sentences[0:2])
        targets.append(file_labels[0:2])
        ids.append((file_sent_ids[0], file_id))

    for i in range(1, len(file_sentences)-1):
        if len(file_sentences[i])+2 > max_len:
            continue
        num_extra = (len(file_sentences[i]) + len(file_sentences[i-1]) + 3) - max_len
        if num_extra > 0:
            inputs.append([file_sentences[i - 1][num_extra:], file_sentences[i]])
            targets.append([file_labels[i - 1][num_extra:], file_labels[i]])
            ids.append((file_sent_ids[i], file_id))
        else:
            inputs.append(file_sentences[i-1:i+2])
            targets.append(file_labels[i-1:i+2])
            ids.append((file_sent_ids[i], file_id))


# get history, respinse data from csv file
def read_file(filename, max_len, train):
    """
     for each example, create a list of words, and a list of tuples as
     labels for each word for each sentence. Create examples that a list of multiple
     sentences,3
    """
    i = 1
    #train = True

    inputs = list()
    targets = list()
    ids = list()

    with open(filename, 'r') as fp:
        data = json.load(fp)

    for file_id, sentences in data.items():
        file_sentences = list()
        file_labels = list()
        file_sent_ids = list()
        for sentence_id, words in sentences.items():
            file_sentences.append([word['word'] for word in words])
            file_labels.append([word['iob_labels'] for word in words])
            file_sent_ids.append(sentence_id)
        if not file_labels:
            continue

        if train:
            make_train_ex(inputs, targets, ids, file_sentences, file_labels, file_sent_ids, max_len, file_id)
        else:
            make_val_ex(inputs, targets, ids, file_sentences, file_labels,
                          file_sent_ids, max_len, file_id)

    return inputs, targets, ids

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
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    SEP_WORD = '<s>'
    EOS_WORD = '</s>'
    CLS_WORD = '<cls>'

    def __init__(self, filename, max_len, vocab=None,
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
        if filename.find("train") != -1:
            self.train = True
        else:
            self.train = False

        if filename is not None:
            self.examples, self.targets, self.ids = read_file(filename, max_len, self.train)

        self.max_len = max_len

        if vocab is None:
            # Create new vocab object
            self.vocab = Vocab(special_tokens=[DialogueDataset.PAD_WORD,
                                               DialogueDataset.UNK_WORD,
                                               DialogueDataset.SEP_WORD,
                                               DialogueDataset.EOS_WORD,
                                               DialogueDataset.CLS_WORD])
        else:
            self.vocab = vocab

        # do not want to update vocab for running old model
        if update_vocab:
            for example in self.examples:
                self.vocab.add_documents(example)

    def _process_input(self, sentences):
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
        for sentence in sentences:
            inputs += sentence
            inputs.append(DialogueDataset.SEP_WORD)
        inputs = inputs[:self.max_len]
        inputs[-1] = DialogueDataset.EOS_WORD


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
        seg = list()
        i = 1
        for j, token in enumerate(inputs):
            if token == self.vocab[DialogueDataset.PAD_WORD]:
                break
            seg.append(i)
            if token == self.vocab[DialogueDataset.SEP_WORD]:
                i += 1
        seg += [0] * needed_pads
        h_seg = np.array(seg, dtype=np.long)

        h_seq = np.array(inputs, dtype=np.long)

        return h_seq, h_pos, h_seg

    def _process_targets(self, targets):
        """
        creates token encodings for the word embeddings, and positional
            encodings for the response

        Examples:
            Response:  <cls> i am good , thank you ! </s>
            self.response_len = 10

            r_seq = np.array([4, 43, 52, 77, 9, 65, 93, 5,  3, 0])
            r_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0,)]

        Args:
            response: list of tokens in the response
        Returns:
            r_seq: token encodings for the response
        """
        default_label = [0,0,0,0,0,0,0,0,0]

        labels = [default_label]
        for target in targets:
            labels += target
            labels.append(default_label)
        labels = labels[:self.max_len]
        labels[-1] = default_label

        needed_pads = self.max_len - len(labels)
        if needed_pads > 0:
            labels = labels + [default_label] * needed_pads

        labels = np.array(labels, dtype=np.long)

        return labels

    def _get_start_end_idx(self, example, ids):
        if self.train is True:
            return (0,0)

        if ids[0] == '0':
            return 1, 1+len(example[0])

        return 2+len(example[0]), 2 + len(example[0]) + len(example[1])

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
        i_seq, i_pos, i_seg = self._process_input(self.examples[index])
        labels = self._process_targets(self.targets[index])
        ids = self.ids[index]
        start_end_idx = self._get_start_end_idx(self.examples[index], ids)
        return i_seq, i_pos, i_seg, labels, ids, start_end_idx

    def __len__(self):
        return len(self.examples)
