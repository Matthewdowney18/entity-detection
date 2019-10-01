from collections import Counter
import numpy as np
import torch.utils.data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer

NUM_IB_LABELS = 5
def make_targets(label):
    if label[NUM_IB_LABELS] == 1:
        return [0] * NUM_IB_LABELS + [1] + [0] * NUM_IB_LABELS

    inside = label[:NUM_IB_LABELS]
    if 1 in inside:
        return inside + [0] * (NUM_IB_LABELS + 1)

    b_label = [0] * NUM_IB_LABELS
    beginning = label[NUM_IB_LABELS+1:]
    for i, label in reversed(list(enumerate(beginning))):
        if label != -1:
            b_label[i] = 1
            break
    return [0] * (NUM_IB_LABELS + 1) + b_label

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
            file_labels.append([make_targets(word['iob_labels']) for word in words])
            file_sent_ids.append(sentence_id)
        if not file_labels:
            continue

        if train:
            make_train_ex(inputs, targets, ids, file_sentences, file_labels, file_sent_ids, max_len, file_id)
        else:
            make_val_ex(inputs, targets, ids, file_sentences, file_labels,
                          file_sent_ids, max_len, file_id)

        #if len(inputs) > 10000:
        #    break

    return inputs, targets, ids


class DialogueDataset(torch.utils.data.Dataset):
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    SEP_WORD = '<s>'
    EOS_WORD = '</s>'
    CLS_WORD = '<cls>'

    def __init__(self, filename, max_len):
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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _process_input(self, sentences):
        """
        creates token encodings for the word embeddings, positional encodings,
        and segment encodings for the dialogue history

        Examples:
            History: <cls> hello ! <s> hi , how are you ? </s>
            self.sentence_len = 15

        Args:
            history: list of tokens in the history
        Returns:
            input_ids
        """

        pad = self.tokenizer.pad_token_id
        sep = self.tokenizer.sep_token_id
        unk = self.tokenizer.unk_token_is
        cla = self.tokenizer.cls_token_id


        inputs = [self.tokenizer.cls_token]
        for sentence in sentences:
            inputs += sentence
            inputs.append(self.tokenizer.sep_token)
        inputs = inputs[:self.max_len]

        needed_pads = self.max_len - len(inputs)
        if needed_pads > 0:
            inputs = inputs + [self.tokenizer.pad_token] * needed_pads

        input_ids = self.tokenizer.convert_tokens_to_ids(inputs)

        input_ids = np.array(input_ids, dtype=np.long)

        return input_ids

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

        default_label = [0,0,0,0,0,1,0,0,0,0,0]

        labels = [default_label]
        for target in targets:
            labels += target
            labels.append(default_label)
        labels = labels[:self.max_len]

        needed_pads = self.max_len - len(labels)
        if needed_pads > 0:
            labels = labels + [default_label] * needed_pads

        labels = [label.index(1) for label in labels]

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
            input ids: segment encoding for the history
            r_seq: token encodings for the response
            r_pos: positional encoding for the response
        """
        input_ids = self._process_input(self.examples[index])
        labels = self._process_targets(self.targets[index])
        ids = self.ids[index]
        start_end_idx = self._get_start_end_idx(self.examples[index], ids)
        return input_ids, labels, ids, start_end_idx

    def __len__(self):
        return len(self.examples)
