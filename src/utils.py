import json
import logging
import re
from difflib import SequenceMatcher

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from thefuzz import fuzz

FALLACIES = [
    "ad_populum",
    "causal_oversimplification",
    "circular_reasoning",
    "equivocation",
    "false_causality",
    "false_dilemma",
    "hasty_generalization",
    "red_herring",
    "slippery_slope",
    "straw_man",
    "appeal_to_anger",
    "appeal_to_fear",
    "appeal_to_positive_emotion",
    "appeal_to_pity",
    "ad_hominem",
    "appeal_to_false_authority",
    "appeal_to_nature",
    "appeal_to_tradition",
    "appeal_to_worse_problems",
    "guilt_by_association",
    "tu_quoque",
]

FALLACIES_TO_LABELS = {
    "ad_populum": 0,
    "causal_oversimplification": 1,
    "circular_reasoning": 2,
    "equivocation": 3,
    "false_causality": 4,
    "false_dilemma": 5,
    "hasty_generalization": 6,
    "red_herring": 7,
    "slippery_slope": 8,
    "straw_man": 9,
    "appeal_to_anger": 10,
    "appeal_to_fear": 11,
    "appeal_to_positive_emotion": 12,
    "appeal_to_pity": 13,
    "ad_hominem": 14,
    "appeal_to_false_authority": 15,
    "appeal_to_nature": 16,
    "appeal_to_tradition": 17,
    "appeal_to_worse_problems": 18,
    "guilt_by_association": 19,
    "tu_quoque": 20,
}


def setup_logger(filename):
    # Create logger
    logger = logging.getLogger("MafaldaLogger")
    logger.setLevel(
        logging.DEBUG
    )  # Set logger level to DEBUG, INFO, WARNING, ERROR, or CRITICAL

    # Create file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)  # Set file handler level

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to file handler
    fh.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(fh)

    return logger


def get_tags(text):
    if not isinstance(text, str):
        return set()
    matches = re.findall(r"<(/?[^>]*)>", text)
    unique_tags = set(tag.strip().replace("/", "") for tag in matches)
    return unique_tags


def build_correction_dict(wrong_tags, correct_tags, threshold=0.8):
    correction_dict = {}

    for wrong_tag in wrong_tags:
        match_ratios = [
            (correct_tag, SequenceMatcher(None, wrong_tag, correct_tag).ratio())
            for correct_tag in correct_tags
        ]
        best_match, best_ratio = max(match_ratios, key=lambda x: x[1])

        if best_ratio > threshold:
            correction_dict[wrong_tag] = best_match

    return correction_dict


def extract_tag_positions(text, row_id):
    pattern = r"<(/?)(.*?)>"
    tags_to_ignore = ["URL", "PHONE_NUMBER", "EMAIL_ADDRESS", "USER"]
    matches = [
        m for m in re.finditer(pattern, text) if m.group(2) not in tags_to_ignore
    ]

    positions = []
    for i in range(0, len(matches), 2):  # assuming each tag has a start and end
        start_tag = matches[i]
        fallacy = start_tag.group(2)
        if fallacy == "no fallacy":
            positions.append([0, 0, fallacy])
            continue
        if i + 1 >= len(matches):
            print(f"Error: missing end tag in row {row_id}")
            break
        end_tag = matches[i + 1]
        # we remove the length of the opening tag and the slash and the first chevron.
        positions.append(
            [start_tag.start(), end_tag.start() - len(fallacy) - 2, fallacy]
        )

    return positions


def chatgpt_annotated_to_jsonl(input_file_path: str) -> None:
    """Transforms a chatgpt annotated file (jsonl, TSV or CSV) to a json line file for Doccano (in the same directory)."""
    if input_file_path.endswith(".jsonl"):
        df = pd.read_json(input_file_path, lines=True)
    elif input_file_path.endswith(".tsv"):
        df = pd.read_csv(input_file_path, sep="\t")
    else:
        df = pd.read_csv(input_file_path)
    df["labels"] = None
    if "Comments" in df.columns:
        df.drop(columns=["Comments"], inplace=True)
    df["Comments"] = None
    all_tags = set()
    for i, row in df.iterrows():
        text = row["annotation"]
        tags = get_tags(text)
        all_tags.update(tags)
        text_wo_tags = remove_tags(text)
        if not isinstance(text_wo_tags, str):
            print(f"WARNING: text is not a string in row {i}.")
            continue
        original_text = row["text"]
        if (
            "no fallacy" not in text_wo_tags
            and abs(len(text_wo_tags) - len(original_text)) > 2
        ):
            print(
                f"WARNING: text length mismatch for row {i} ({len(text_wo_tags)} != {len(row['text'])})."
            )
        df.at[i, "text"] = (
            original_text if "no fallacy" in text_wo_tags else text_wo_tags
        )
        df.at[i, "Comments"] = [row["explanation"]]
        spans = extract_tag_positions(text, i)
        df.at[i, "labels"] = spans
    correction_mapping = build_correction_dict(all_tags, FALLACIES)
    for i, row in df.iterrows():
        spans = row["labels"]
        if spans is None:
            print(f"WARNING: no spans in row {i}.")
            continue
        for span in spans:
            span[2] = correction_mapping.get(span[2], span[2])
            if span[0] == span[1]:
                span[0] = 0
                span[1] = len(row["text"]) - 1
            if span[2] not in FALLACIES:
                span[2] = "JOCKER"
        df.at[i, "labels"] = spans
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True)
    df.drop(columns=["annotation", "explanation", "Unnamed: 0"], inplace=True)
    df.to_json(input_file_path.replace(".tsv", ".jsonl"), orient="records", lines=True)


def read_jsonl(file_path: str) -> list[dict]:
    """Reads a JSONL file into a list of dicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f]


def extract_fallacy(input_text):
    # Initialize output dictionary
    output = {fallacy: [] for fallacy in FALLACIES}

    # Regex pattern to match tags
    pattern = r"<([a-zA-Z_]+)>(.*?)</([a-zA-Z_]+)"
    matches = re.findall(pattern, input_text, re.DOTALL)

    # Mapping for special cases
    special_cases = {
        "appeal_to_authority": "appeal_to_false_authority",
        "circul_reasoning": "circular_reasoning",
        "circ_reasoning": "circular_reasoning",
    }

    # Ignore fallacies
    ignore_fallacies = ["causal_claim", "no_fallacy"]

    for tag_open, content, tag_close in matches:
        # Check for invalid tags
        if tag_open != tag_close or "<" in content or ">" in content:
            continue

        # Adjust fallacy name
        fallacy = tag_open
        if fallacy.startswith("fallacy_"):
            fallacy = fallacy[8:]

        fallacy = special_cases.get(fallacy, fallacy)

        # Check for ignore fallacies and valid fallacies
        if fallacy in ignore_fallacies or fallacy not in FALLACIES:
            continue

        assert fallacy in FALLACIES, f"Fallacy {fallacy} not in FALLACIES"
        # Add to output
        output[fallacy].append(content)

    return output


def remove_tags(input_text):
    for fallacy in FALLACIES:
        pattern_open = r"<fallacy_{0}>".format(fallacy)
        pattern_close = r"</fallacy_{0}>".format(fallacy)
        input_text = re.sub(pattern_open, "", input_text)
        input_text = re.sub(pattern_close, "", input_text)

    return input_text


def match_labels_with_text_jsonl(jsonl_file):
    data = read_jsonl(jsonl_file)
    list_outputs = []
    for i in data:
        output = {}
        for fallacy in FALLACIES:
            output[fallacy] = []
        text = i["text"]
        labels = i["label"]
        for l in labels:
            label = l[2]
            if "hope" in label:
                label = "appeal_to_positive_emotion"
            label = label.replace(" ", "_")
            label = label.replace("(", "")
            label = label.replace(")", "")
            output[label].append(text[l[0] : l[1]])

        list_outputs.append(output)

    return list_outputs


def split_dataset(data_path):
    data = pd.read_csv(data_path, sep="\t")
    data["id"] = [f"f_{i}" for i in range(len(data))]
    data = data.reindex(
        columns=["id", "text", "tokens", "source_dataset", "text_length", "fallacies"]
    )
    train, test = train_test_split(data, test_size=0.12, random_state=42)
    train.to_csv("datasets/train.tsv", sep="\t")
    test.to_csv("datasets/test.tsv", sep="\t")
    print("Train size:", len(train))
    print("Test size:", len(test))


def remove_end_punctuation(text):
    if text[-1] == '"':
        text = text[:-1]
    if text[-1] in [".", ",", "!", "?"]:
        text = text[:-1]
    return text


def find_unique_tags(input_file_path):
    if input_file_path.endswith(".jsonl"):
        data = pd.read_json(input_file_path, lines=True)
    elif input_file_path.endswith(".tsv"):
        data = pd.read_csv(input_file_path, sep="\t")
    else:
        data = pd.read_csv(input_file_path)
    set_tags = set()

    for _, row in data.iterrows():
        annotated_text = str(row["annotation"])
        tag_pattern = r"\<\/(.*?)\>"
        match = re.findall(tag_pattern, annotated_text)
        for m in match:
            set_tags.update([m])
    return set_tags


def parse_paragraph_window(paragraph, window_size=3):
    sentences = nltk.sent_tokenize(paragraph)
    n_gram_sentences = list(zip(*[sentences[i:] for i in range(window_size)]))
    return n_gram_sentences


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def fix_punctuation_spacing(text):
    # This pattern matches any punctuation character (.,!?;:) which has a whitespace character before it
    pattern = r"\s([.,!?;:\'\"])"
    # Replace the matched pattern with the punctuation without a space before it
    fixed_text = re.sub(pattern, r"\1", text)
    return fixed_text


def find_best_text_part(query, text):
    n_grams = len(query.split())
    tokenized_text = text.split()
    list_possible_text = []
    beg = 0

    if n_grams - 4 > 0:
        beg = n_grams - 4

    for i in range(beg, n_grams + 4):
        for ngram in find_ngrams(tokenized_text, i):
            sub_text = " ".join(ngram)
            sub_text = fix_punctuation_spacing(sub_text)

            ratio = fuzz.ratio(query, sub_text)

            if ratio > 80:
                list_possible_text.append([ratio, sub_text])

    if len(list_possible_text) > 0:
        return max(list_possible_text, key=lambda x: x[0])[1]
    else:
        return None


def transform_to_doccano(input_file_path):
    if input_file_path.endswith(".jsonl"):
        data = pd.read_json(input_file_path, lines=True)
    elif input_file_path.endswith(".tsv"):
        data = pd.read_csv(input_file_path, sep="\t")
    else:
        data = pd.read_csv(input_file_path)

    with open("doccano_input.jsonl", "w") as f:
        cnt_not_found = 0
        list_best_text_not_found = []
        all_tags = find_unique_tags(input_file_path)
        cnt_all = 0
        cnt_found = 0

        for _, row in data.iterrows():
            annotated_text = str(row["annotation"])
            unannotated_text = str(row["text"])
            unannotated_text = unannotated_text.strip()
            annotated_text = annotated_text.replace("\r\n", " ")
            unannotated_text = unannotated_text.replace("\r\n", " ")
            annotated_text = fix_punctuation_spacing(annotated_text)
            unannotated_text = fix_punctuation_spacing(unannotated_text)
            unannotated_text = " ".join(unannotated_text.split())

            fallacies = {}
            for fallacy in all_tags:
                tag_pattern = r"\<{0}\>([\s\S]*?)\<\/{0}\>".format(fallacy)
                match = re.findall(tag_pattern, annotated_text)

                if len(match) > 0:
                    fallacies[fallacy] = []

                    for m in match:
                        m = remove_tags(m)
                        m = remove_end_punctuation(m)
                        m = m.strip()
                        if m != "" and m != "..":
                            cnt_all += 1
                            start_idx = unannotated_text.find(m)
                            if start_idx != -1:
                                end_idx = start_idx + len(m)
                                fallacies[fallacy].append([start_idx, end_idx])
                                cnt_found += 1
                            else:
                                best_text_part = find_best_text_part(
                                    m, unannotated_text
                                )
                                if best_text_part is not None:
                                    start_idx = unannotated_text.find(best_text_part)
                                    end_idx = start_idx + len(best_text_part)
                                    fallacies[fallacy].append([start_idx, end_idx])
                                    if start_idx == -1:
                                        print(best_text_part)
                                        print(unannotated_text)
                                        print("ERROR")
                                    cnt_found += 1
                                else:
                                    cnt_not_found += 1
                                    list_best_text_not_found.append(
                                        [m, unannotated_text]
                                    )
            labels = []
            for fallacy in fallacies:
                for pos in fallacies[fallacy]:
                    if "authority" in fallacy:
                        fallacy = "appeal to (false) authority"
                    labels.append([pos[0], pos[1], fallacy.replace("_", " ")])

            json_line = json.dumps(
                {
                    "text": unannotated_text
                    + "\nEXPLANATION:\n"
                    + str(row["explanation"]),
                    "labels": labels,
                }
            )
            f.write(json_line + "\n")
        print(cnt_not_found)
        print(list_best_text_not_found)
        print(cnt_found)
        print(cnt_all)


def non_fallacious_data_to_doccano(data_path: str, key: str, limit: int):
    data = pd.read_csv(data_path, sep="\t")
    data = data.sample(frac=1)
    data = data.reset_index()
    with open("doccano_input.jsonl", "w") as f:
        for id, row in data.iterrows():
            json_line = json.dumps(
                {
                    "text": row[key],
                    "labels": [],
                }
            )
            f.write(json_line + "\n")
            if id == limit:
                break


if __name__ == "__main__":
    ex = "Hello, how are you doing? Today is a beautiful day! It's great to see you. Let's have some fun. Hello world."
    n_gram_sentences = parse_paragraph_window(ex, 3)
    print(n_gram_sentences)
