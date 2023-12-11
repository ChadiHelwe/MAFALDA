import json
import re
from collections import OrderedDict

from src.utils import read_jsonl


def split_paragraph_into_sentences(paragraph):
    # Define a regular expression pattern for sentence splitting
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    # Use the re.split() function to split the paragraph into sentences
    sentences = re.split(sentence_pattern, paragraph)

    return sentences


def find_non_fallacies_indexes(tuples, start_index, end_index):
    non_fallacies_indexes = []
    sorted_tuples = sorted(tuples, key=lambda x: x[0])

    if len(sorted_tuples) == 0:
        return {(start_index, end_index): ["nothing"]}

    if sorted_tuples[0][0] > start_index:
        non_fallacies_indexes.append((start_index, sorted_tuples[0][0] - 1))

    current_end = sorted_tuples[0][1]

    for start, end, _ in sorted_tuples[1:]:
        if start > current_end + 1:
            non_fallacies_indexes.append((current_end + 1, start - 1))
        current_end = max(current_end, end)

    if current_end < end_index:
        non_fallacies_indexes.append((current_end + 1, end_index))

    dict_non_fallacies_indexes = {}
    for i in range(len(non_fallacies_indexes)):
        if (non_fallacies_indexes[i][1] - non_fallacies_indexes[i][0]) > 10:
            dict_non_fallacies_indexes[
                (non_fallacies_indexes[i][0], non_fallacies_indexes[i][1])
            ] = ["nothing"]

    return dict_non_fallacies_indexes


def merge_all_labels(non_fallacious_labels, fallacious_labels):
    all_labels = {}
    for label in non_fallacious_labels:
        all_labels[label] = non_fallacious_labels[label]
    for label in fallacious_labels:
        all_labels[label] = fallacious_labels[label]
    sorted_all_labels = sorted(all_labels.items(), key=lambda x: x[0][0])
    dict_sorted_all_labels = OrderedDict()
    for label in sorted_all_labels:
        dict_sorted_all_labels[label[0]] = label[1]
    return dict_sorted_all_labels


def process_data(data):
    all_examples = []
    for example in data:
        text = example["text"]
        fallacious_labels = example["labels"]
        non_fallacious_labels = find_non_fallacies_indexes(
            fallacious_labels, 0, len(text)
        )

        or_and_fallacious_labels = {}
        split_paragraph = OrderedDict()

        for label in fallacious_labels:
            if label[2].lower() != "to clean":
                if (label[0], label[1]) not in or_and_fallacious_labels:
                    or_and_fallacious_labels[(label[0], label[1])] = [label[2].lower()]
                else:
                    or_and_fallacious_labels[(label[0], label[1])].append(
                        label[2].lower()
                    )

        or_and_labels = merge_all_labels(
            non_fallacious_labels, or_and_fallacious_labels
        )
        for or_labels in or_and_labels:
            extracted_labeled_text = text[or_labels[0] : or_labels[1]]
            split_extracted_labeled_text = split_paragraph_into_sentences(
                extracted_labeled_text
            )
            # print(or_labels)
            # print(extracted_labeled_text)
            # print(split_extracted_labeled_text)
            # print(split_paragraph)
            for extracted_text in split_extracted_labeled_text:
                if extracted_text != "":
                    if extracted_text not in split_paragraph:
                        split_paragraph[extracted_text] = [
                            set(or_and_labels[or_labels])
                        ]
                    else:
                        split_paragraph[extracted_text].append(
                            set(or_and_labels[or_labels])
                        )
        all_examples.append(split_paragraph)

    return all_examples
    # index_labels = [idx for idx in or_and_labels]
    # extract_non_fallacies_indexes(index_labels)


def generate_testing_set(data_path):
    data = read_jsonl(data_path)
    processed_data = process_data(data)
    with open("datasets/post_process_final_gold_standard_dataset.jsonl", "w") as f:
        for example, processed_example in zip(data, processed_data):
            # sentences_w_labels = "{"
            # for sentence in processed_example:
            #     sentences_w_labels += f'"{sentence}": {processed_example[sentence]},'
            # sentences_w_labels = sentences_w_labels[:-1] + "}"
            sentences_w_labels = {}
            tmp_processed_example = {}
            for sentence in processed_example:
                for label in processed_example[sentence]:
                    if sentence not in tmp_processed_example:
                        tmp_processed_example[sentence] = [list(label)]
                    else:
                        tmp_processed_example[sentence].append(list(label))
                sentences_w_labels[sentence] = tmp_processed_example[sentence]

            json_line = json.dumps(
                {
                    "text": example["text"],
                    "labels": example["label"],
                    "sentences_with_labels": json.dumps(sentences_w_labels),
                }
            )
            f.write(json_line + "\n")
