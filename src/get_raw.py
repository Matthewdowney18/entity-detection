import json

filename = "reformatted_data/data_6/train_data.json"

with open(filename, 'r') as fp:
    data = json.load(fp)


all_data = str()
for file_id, sentences in data.items():
    all_sentences = list()
    all_labels = list()
    for sentence_id, words in sentences.items():
        all_data += " ".join(word['word'] for word in words)
        all_data += " "
    all_data += "\n"

with open("reformatted_data/raw_2.txt", 'w') as f:
    f.write(all_data)
