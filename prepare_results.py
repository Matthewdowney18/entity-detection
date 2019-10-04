import json
import os

def main():
    filename = "experiments/exp_3/run_5/predictions50.json"

    with open(filename, 'r') as f:
        data = json.load(f)

    target_lines = list()
    response_lines = list()
    for doc_id, sentences in data.items():
        if doc_id == "results":
            continue
        file_id = doc_id.split('-')[0]
        doc_num = doc_id.split('-')[1]
        header = "#begin document ({}); part {:0>3d}\n".format(file_id, int(doc_num))
        target_lines.append(header)
        response_lines.append(header)
        for sentence, words in sentences.items():
            if "prediction_str" not in words[0]:
                continue
            for word in words:
                t_line = [file_id, doc_num, str(word["word_nb"]),
                          word["coref_str"]]
                r_line = [file_id, doc_num, str(word["word_nb"]),
                          word["prediction_str"].replace('*', '0')]
                target_lines.append(" ".join(t_line) + "\n")
                response_lines.append(" ".join(r_line) + "\n")
        target_lines.append("\n#end documents\n")
        response_lines.append("\n#end documents\n")

    output_dir = "/home/mattd/PythonProjects/coreference/reference-coreference-scorers/"
    with open(os.path.join(output_dir, "response.txt"), 'w') as f:
        f.writelines(response_lines)

    with open(os.path.join(output_dir, "target.txt"), 'w') as f:
        f.writelines(target_lines)


if __name__=='__main__':
    main()