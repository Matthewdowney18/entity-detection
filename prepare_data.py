import argparse
import subprocess
import json
import os
import time
import pandas as pd
import multiprocessing
import sys
import re
import random
from tqdm import tqdm
import time

def get_labels(string, num_inside, inside_ids):
    n = 0
    outside = [0, 0, 0, 0]
    beginning = [0, 0, 0, 0]
    o_b_ids = [-1, -1, -1, -1, -1, -1, -1, -1]
    inside_ids = list() + inside_ids

    # is an inside token if num_inside is not zero
    inside = int(num_inside > 0)

    # def: mention string - "(34", "(45)", "43)"
    # split string into mention strings, process each one seperately

    mention_strings = string.split('|')

    if len(mention_strings) == 1:
        if mention_strings[0] == "-":
            return [inside] + [0,0,0,0,0,0,0,0], o_b_ids, num_inside, inside_ids

    for i, mention_string in enumerate(mention_strings):
        # this is incase there are too many mentions to keep track of
        if i > 3:
            if re.search('\d+\)', mention_string):
                coref_id = int(mention_string[:-1])
                inside_ids.remove(coref_id)
                num_inside -= 1
            elif re.search('\(\d+', mention_string):
                coref_id = int(mention_string[1:])
                num_inside += 1
                inside_ids.append(coref_id)
            break

        # beginning and outside
        if re.search('\(\d+\)', mention_string):
            coref_id = int(mention_string[1:-1])
            outside[n] = 1
            beginning[n] = 1
            o_b_ids[n + 4] = coref_id
            o_b_ids[n] = coref_id
            n+=1

        # just outside
        elif re.search('\d+\)', mention_string):
            coref_id = int(mention_string[:-1])
            outside[n] = 1
            o_b_ids[n] = coref_id
            n+=1
            num_inside -= 1
            inside_ids.remove(coref_id)

        # just beginning
        elif re.search('\(\d+', mention_string):
            coref_id = int(mention_string[1:])
            beginning[n] = 1
            o_b_ids[n + 4] = coref_id
            n += 1
            num_inside += 1
            inside_ids.append(coref_id)

    # if it creates the mention, then it is not an inside token. so only set
    # as inside if it was inside to begin with, and it is still inside
    inside = int(inside and num_inside > 0)

    return [inside] + outside + beginning, o_b_ids, num_inside, inside_ids


def create_dict_from_df(df):


    data = dict()
    document = dict()
    sentence = list()
    num_inside = 0
    inside_ids = list()
    sent_id = 0
    doc_id = df.iloc[0]['doc_id']
    start = True

    for index, row in tqdm(df.iterrows()):
        #if row["file_id"] == "nw/wsj/11/wsj_1121":
        #    print("yo")
        # save sentence to document when new sentence is starting
        # increment sentence id
        if row['word_nb'] == 0 and not start:
            document[sent_id] = sentence
            sentence = list()
            sent_id += 1


        # save document, if a new document, reset sentence id
        if row['doc_id'] != doc_id:
            data[doc_id] = document
            doc_id = row['doc_id']
            document = dict()
            sent_id = 0

        labels, o_b_ids, num_inside, inside_ids = get_labels(row['coref'],
                                                             num_inside,
                                                             inside_ids)
        word = {'word': row['word'],
                'pos': row['pos'],
                'parse': row['parse'],
                'lemma': row['predicate_lemma'],
                'frame': row['predicate_frame'],
                'word_sense': row['word_sense'],
                'name_entities': ['name_entities'],
                'word_nb': row['word_nb'],
                'speaker': row['speaker'],
                'iob_labels': labels,
                'inside_ids': inside_ids,
                'num_inside': num_inside,
                'o_b_ids': o_b_ids}
        sentence.append(word)
        start = False
    return data


def get_df(data_file, dataFrame=None, n_fields=12):
    """ function from yuanliangs coreference model"""
    data_list = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                fields = line.split()
                try:
                    assert len(fields) >= n_fields
                except AssertionError:
                    print(fields)
                    raise AssertionError
                fields = fields[:11] + [fields[-1]]
                data_list.append(fields)
                # df.loc[len(df)] = fields

    if not data_list:
        return None

    # 12 columns, ignore predicate arguments
    columns = ['file_id', 'part_nb', 'word_nb', 'word', 'pos', 'parse', 'predicate_lemma',
               'predicate_frame', 'word_sense', 'speaker', 'name_entities', 'coref']

    new_data = pd.DataFrame(data_list, columns=columns)
    unique_doc_ids = new_data["file_id"] + '-' + new_data["part_nb"].map(str)
    new_data.insert(loc=0, column='doc_id', value=unique_doc_ids)
    new_data = new_data.drop(['part_nb'], axis=1)

    if dataFrame is None:
        dataFrame = new_data

    else:
        dataFrame = dataFrame.append(new_data)

    return dataFrame


def build_dataFrame(path, threads=1, suffix='gold_conll'):
    """ function from yuanliangs coreference model"""
    def worker(pid):
        print("worker %d started..." % pid)
        df = None
        counter = 0
        while not file_queue.empty():
            data_file = file_queue.get()
            # sys.stdout.write("Worker %d: %d files remained to be processed\r" % (pid, file_queue.qsize()))
            df = get_df(data_file, dataFrame=df)
            counter += 1
            if df is not None and counter % 10 == 0:
                data_queue.put(df)
                df = None
        if df is not None:
            data_queue.put(df)
        print("\nWorker %d closed." % pid)

    def worker_alive(workers):
        worker_alive = False
        for p in workers:
            if p.is_alive(): worker_alive = True
        return worker_alive

    assert os.path.isdir(path)
    cmd = 'find ' + path + ' -name "*%s"' % suffix
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,stdin=subprocess.PIPE)
    file_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue(maxsize=10)
    for item in proc.stdout:
        file_queue.put(item.strip())
    n_files = file_queue.qsize()
    print('%d conll files found in %s' % (n_files, path))

    workers = [multiprocessing.Process(target=worker, args=(pid,)) for pid in range(threads)]

    for p in workers:
        p.daemon = True
        p.start()

    time.sleep(1)
    df = None

    while df is None or len(df.file_id.unique()) < n_files:
        item = data_queue.get()
        if df is None:
            df = item
        else:
            df = df.append(item)
        sys.stdout.write("Processed %d files from data queue\r" % len(df.file_id.unique()))
        # if not worker_alive(workers): break

    time.sleep(1)
    assert data_queue.empty()
    # Exit the completed processes
    print("\nFinished assembling data frame.")
    for p in workers:
        p.join()

    # df.part_nb = pd.to_numeric(df.part_nb, errors='coerce')
    df.word_nb = pd.to_numeric(df.word_nb, errors='coerce')
    print("\ndata frame is built successfully!")
    print("Processed files: %d" % len(df.file_id.unique()))

    return df


def save_data(all_data, args, meta_data):
    keys = list(all_data.keys())
    meta_data["num_files"] = len(keys)
    random.shuffle(keys)

    train_data = {key: all_data[key] for key in keys[:int(.8 * len(keys))]}
    val_data = {key: all_data[key] for key in
                keys[int(.8 * len(keys)): -int(.1 * len(keys))]}
    test_data = {key: all_data[key] for key in keys[-int(.1 * len(keys)):]}

    with open(os.path.join(args.output_dir, "train_data.json"), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(args.output_dir, "val_data.json"), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(args.output_dir, "test_data.json"), 'w') as f:
        json.dump(test_data, f)

    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(meta_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_filename",
                        default="/home/mattd/datasets/conll2012/train_v4/v4/data/train/data/english/annotations",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the conll files for the task.")
    parser.add_argument("-output_dir",
                        default="reformatted_data/data_8",
                        type=str,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-sources",
                        default=['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb'],
                        type=list,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-type",
                        default="none_original",
                        type=str,
                        required=False,
                        help="The genres you would like to use.")
    parser.add_argument("-max_history_tokens",
                        default=100,
                        type=int,
                        help="the maximum amout of history tokens")
    parser.add_argument("-a_nice_note",
                        default="only dialogues 1-10",
                        type=str,
                        required=False,
                        help="gimme allllll of it")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    meta_data = dict()
    meta_data["args"] = dict()

    for arg in vars(args):
        meta_data["args"][arg] = getattr(args, arg)

    all_data = dict()
    first = True

    # load from each of the files in conll
    for file_1 in args.sources:
        for file_2 in os.listdir(os.path.join(args.dataset_filename, file_1)):
            for file_3 in os.listdir(os.path.join(args.dataset_filename, file_1, file_2)):
                path = "{}/{}/{}/".format(file_1, file_2, file_3)
                print("loading from: {}".format(path))
                df = build_dataFrame(os.path.join(args.dataset_filename, path))
                data = create_dict_from_df(df)
                all_data.update(data)

    save_data(all_data, args, meta_data)

    with open(os.path.join(args.output_dir, "all_data.json"), 'w') as f:
        json.dump(all_data, f)




if __name__ == '__main__':
    main()