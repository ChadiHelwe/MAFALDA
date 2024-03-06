import csv
import math
import os
import re
import traceback
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from src.metrics import (
    AnnotatedText,
    GroundTruthSpan,
    NbClasses,
    PredictionSpan,
    text_label_only_p_r_f1,
)
from src.new_metrics import text_full_task_p_r_f1
from src.utils import read_jsonl

LEVEL_0_FALLACIES_REGEX = r"\b(?:contains|yes|true|is a fallacy|is part)\b"
NON_FALLACIES_REGEX = r"\b(?:does not contain|no|none|not|false|nothing|is not part|not necessarily part|no fallacious|not fallacious)\b"


KEWORDS_LEVEL_1_NUMERIC = {
    "emotion": 1,
    "logic": 2,
    "credibility": 3,
}

LEVEL_2_TO_LEVEL_1 = {
    "nothing": 0,
    "appeal to positive emotion": 1,
    "appeal to anger": 1,
    "appeal to fear": 1,
    "appeal to pity": 1,
    "appeal to ridicule": 1,
    "appeal to worse problems": 1,
    "causal oversimplification": 2,
    "circular reasoning": 2,
    "equivocation": 2,
    "false analogy": 2,
    "false causality": 2,
    "false dilemma": 2,
    "hasty generalization": 2,
    "slippery slope": 2,
    "straw man": 2,
    "fallacy of division": 2,
    "ad hominem": 3,
    "ad populum": 3,
    "appeal to (false) authority": 3,
    "appeal to nature": 3,
    "appeal to tradition": 3,
    "guilt by association": 3,
    "tu quoque": 3,
}


LEVEL_2_NUMERIC = {
    "nothing": 0,
    "appeal to positive emotion": 1,
    "appeal to anger": 2,
    "appeal to fear": 3,
    "appeal to pity": 4,
    "appeal to ridicule": 5,
    "appeal to worse problems": 6,
    "causal oversimplification": 7,
    "circular reasoning": 8,
    "equivocation": 9,
    "false analogy": 10,
    "false causality": 11,
    "false dilemma": 12,
    "hasty generalization": 13,
    "slippery slope": 14,
    "straw man": 15,
    "fallacy of division": 16,
    "ad hominem": 17,
    "ad populum": 18,
    "appeal to (false) authority": 19,
    "appeal to nature": 20,
    "appeal to tradition": 21,
    "guilt by association": 22,
    "tu quoque": 23,
}


NUMERIC_TO_LEVEL_2 = {
    0: "nothing",
    1: "appeal to positive emotion",
    2: "appeal to anger",
    3: "appeal to fear",
    4: "appeal to pity",
    5: "appeal to ridicule",
    6: "appeal to worse problems",
    7: "causal oversimplification",
    8: "circular reasoning",
    9: "equivocation",
    10: "false analogy",
    11: "false causality",
    12: "false dilemma",
    13: "hasty generalization",
    14: "slippery slope",
    15: "straw man",
    16: "fallacy of division",
    17: "ad hominem",
    18: "ad populum",
    19: "appeal to (false) authority",
    20: "appeal to nature",
    21: "appeal to tradition",
    22: "guilt by association",
    23: "tu quoque",
    24: "unknown",
}

KEYWORDS_LEVEL_2_NUMERIC = {
    "emotion": 1,
    "anger": 2,
    "fear": 3,
    "pity": 4,
    "ridicule": 5,
    "worse": 6,
    "problems": 6,
    "oversimplification": 7,
    "circular": 8,
    "equivocation": 9,
    "analogy": 10,
    "causality": 11,
    "dilemma": 12,
    "generalization": 13,
    "slippery": 14,
    "slope": 14,
    "straw": 15,
    "division": 16,
    "hominem": 17,
    "populum": 18,
    "authority": 19,
    "nature": 20,
    "tradition": 21,
    "association": 22,
    "quoque": 23,
}


LEVEL_2_TO_1 = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3,
    22: 3,
    23: 3,
    24: 4,
}


FALLACIES_LEVEL_2_TO_LEVEL_1 = {
    "nothing": "nothing",
    "appeal to positive emotion": "emotion",
    "appeal to anger": "emotion",
    "appeal to fear": "emotion",
    "appeal to pity": "emotion",
    "appeal to ridicule": "emotion",
    "appeal to worse problems": "emotion",
    "causal oversimplification": "logic",
    "circular reasoning": "logic",
    "equivocation": "logic",
    "false analogy": "logic",
    "false causality": "logic",
    "false dilemma": "logic",
    "hasty generalization": "logic",
    "slippery slope": "logic",
    "straw man": "logic",
    "fallacy of division": "logic",
    "ad hominem": "credibility",
    "ad populum": "credibility",
    "appeal to (false) authority": "credibility",
    "appeal to nature": "credibility",
    "appeal to tradition": "credibility",
    "guilt by association": "credibility",
    "tu quoque": "credibility",
    "unknown": "unknown",
}

MODELS_INSTUCTIONS_TAGS = {
    "LLaMA2": ("[INST]", "[/INST]"),
    "LLaMA2-Chat": ("[INST]", "[/INST]"),
    "LLaMA2-Instruct": ("[INST]", "[/INST]"),
    "Falcon": ("[INST]", "[/INST]"),
    "Mistral": ("<s> Instruction:", ""),
    "Mistral-Instruct": ("<s>[INST]", "[/INST]"),
    "Vicuna": ("<s>[INST]", "[/INST]"),
    "WizardLM": ("USER:", "ASSISTANT:"),
    "Zephyr": ("<|system|> </s> <|user|>", "</s> <|assistant|>"),
}


def clean_pred_output(pred_output: str):
    return re.sub(r"[^\w\s]", " ", pred_output)


def extract_labels_level_2(pred_output: str):
    idx_output = pred_output.find("Output:")
    if idx_output != -1:
        pred_output = pred_output[idx_output:]
    idx_to_remove = pred_output.find("Based on the above text, determine")
    if idx_to_remove != -1:
        # the model has repetead the instructions
        pred_output = pred_output[:idx_to_remove]
    pred_output = clean_pred_output(pred_output.lower())
    instruction_idx = pred_output.find("based on the above")
    if instruction_idx != -1:
        pred_output = pred_output[:instruction_idx]
    set_labels = set()
    for keyword in KEYWORDS_LEVEL_2_NUMERIC:
        if keyword in pred_output:
            set_labels.add(KEYWORDS_LEVEL_2_NUMERIC[keyword])

    # print(str(pred_output).lower())
    if re.search(NON_FALLACIES_REGEX, str(pred_output).lower()):
        set_labels.add(0)
    if set_labels == set():
        set_labels.add(24)

    return set_labels


def extract_labels_level_1(pred_output: str):
    idx_output = pred_output.find("Output:")
    if idx_output != -1:
        pred_output = pred_output[idx_output:]
    idx_to_remove = pred_output.find("Based on the above text, determine")
    if idx_to_remove != -1:
        # the model has repetead the instructions
        pred_output = pred_output[:idx_to_remove]
    pred_output = clean_pred_output(pred_output.lower())
    instruction_idx = pred_output.find("based on the above")
    if instruction_idx != -1:
        pred_output = pred_output[:instruction_idx]
    set_labels = set()
    for keyword in KEWORDS_LEVEL_1_NUMERIC:
        if keyword in pred_output:
            set_labels.add(KEWORDS_LEVEL_1_NUMERIC[keyword])

    # print(str(pred_output).lower())
    if re.search(NON_FALLACIES_REGEX, str(pred_output).lower()):
        set_labels.add(0)
    if set_labels == set():
        set_labels.add(4)

    return set_labels


def __concatenate_sentences_to_spans(
    sentences_with_labels: OrderedDict[str, set[Any]]
) -> AnnotatedText:
    spans = []
    list_sentences_with_labels = list(sentences_with_labels.keys())
    # print(sentences_with_labels)
    begin_index = 0
    # Iterate over all sentences
    for i in range(0, len(list_sentences_with_labels)):
        found_next_sentence = False
        current_sentence = list_sentences_with_labels[
            i
        ]  # Current sentence being considered
        current_labels = sentences_with_labels[
            current_sentence
        ]  # Labels associated with the current sentence

        # Iterate over the labels associated with the current sentence
        for label in current_labels:
            tmp_span = (
                current_sentence  # Initialize temporary span with the current sentence
            )

            # Check following sentences for the same label to concatenate
            for j in range(i + 1, len(list_sentences_with_labels)):
                found_next_sentence = True
                next_sentence = list_sentences_with_labels[
                    j
                ]  # Next sentence to consider
                next_labels = sentences_with_labels[
                    next_sentence
                ]  # Labels for the next sentence

                # If the same label is found, concatenate the sentence
                if label in next_labels:
                    tmp_span += " " + next_sentence
                    sentences_with_labels[next_sentence].remove(
                        label
                    )  # Remove label to avoid duplication
                else:
                    # If the label is not in the next sentences, finalize the span
                    end_index = begin_index + len(tmp_span)
                    spans.append(
                        PredictionSpan(tmp_span, label, [begin_index, end_index])
                    )
                    break
                # Edge case: if we're at the last sentence and it has the same label, finalize the span
                if j == len(list_sentences_with_labels) - 1:
                    end_index = begin_index + len(tmp_span)
                    spans.append(
                        PredictionSpan(tmp_span, label, [begin_index, end_index])
                    )
            if not found_next_sentence:
                end_index = begin_index + len(tmp_span)
                spans.append(PredictionSpan(tmp_span, label, [begin_index, end_index]))

        begin_index += len(current_sentence) + 1  # Check + 1

    return AnnotatedText(spans)


def build_ground_truth_spans(text: str, labels: list[list[Any]]):
    dict_labels = OrderedDict()
    for label in labels:
        if "to clean" in label[2].lower():
            continue
        if (label[0], label[1]) not in dict_labels:
            dict_labels[(label[0], label[1])] = set([LEVEL_2_NUMERIC[label[2].lower()]])
        else:
            dict_labels[(label[0], label[1])].add(LEVEL_2_NUMERIC[label[2].lower()])

    current = 0
    end = len(text)
    uncovered_ranges = []

    # Find and store ranges of text that are not covered
    for idx in dict_labels:
        # If there is a gap before the current labeled span, add it as uncovered
        if current < idx[0]:
            uncovered_ranges.append((current, idx[0] - 1))

        # Update the current index to the end of the labeled span
        current = max(current, idx[1] + 1)

    # If there is any remaining text after the last label, add it as uncovered
    if current < end:
        uncovered_ranges.append((current, end))

    # If there were no labels at all, the entire text is uncovered
    if len(dict_labels) == 0:
        uncovered_ranges.append((0, end))

    # Add uncovered ranges to the dictionary with a None labe
    for i in uncovered_ranges:
        dict_labels[i] = set([None])

    # Construct the list of GroundTruthSpan objects
    ground_truth_spans = []

    for idx in dict_labels:
        # Create a GroundTruthSpan for each labeled and uncovered span
        ground_truth_spans.append(
            GroundTruthSpan(text[idx[0] : idx[1]], dict_labels[idx], [idx[0], idx[1]])
        )

    return AnnotatedText(ground_truth_spans)


def build_prediction_spans(
    pred_dataset,
    gold_dataset,
    begin_instruction_tag="",
    end_instruction_tag="",
    level=2,
):
    all_y_pred = []

    for _, pred_instance in zip(gold_dataset, pred_dataset):
        # text_sentences = list(json.loads(true_instance["sentences_with_labels"]).keys())
        y_pred = OrderedDict()

        for sentence, generated_out in pred_instance["prediction"].items():
            if level == 2:
                pred_label = extract_labels_level_2(generated_out)
            else:
                pred_label = extract_labels_level_1(generated_out)
            y_pred[sentence] = pred_label

        all_y_pred.append(__concatenate_sentences_to_spans(y_pred))

    return all_y_pred


def concatenate_sentences_to_spans_levels(y_pred: [OrderedDict], level: int = 2):  # type: ignore
    all_y_pred = []
    for y_pred_instance in y_pred:
        if level == 1:
            for sentence in y_pred_instance:
                y_pred_instance[sentence] = set(
                    [LEVEL_2_TO_1[i] for i in y_pred_instance[sentence]]
                )
        elif level == 0:
            for sentence in y_pred_instance:
                tmp_labels = set()
                for i in y_pred_instance[sentence]:
                    if i == 0:
                        tmp_labels.add(0)
                    elif i >= 24:
                        tmp_labels.add(0)
                    else:
                        tmp_labels.add(1)
                y_pred_instance[sentence] = tmp_labels
        y_pred_instance = __concatenate_sentences_to_spans(y_pred_instance)
        all_y_pred.append(y_pred_instance)
    return all_y_pred


def evaluate(dataset_path: str, prediction_path: str):
    all_y_true = []

    gold_dataset = read_jsonl(dataset_path)
    pred_dataset = read_jsonl(prediction_path)

    begin_instruction_tag = ""
    end_instruction_tag = ""

    for model in MODELS_INSTUCTIONS_TAGS:
        if model in prediction_path:
            begin_instruction_tag = MODELS_INSTUCTIONS_TAGS[model][0]
            end_instruction_tag = MODELS_INSTUCTIONS_TAGS[model][1]
            break

    # Build ground truth spans for each instance in the gold dataset
    for i in gold_dataset:
        all_y_true.append(build_ground_truth_spans(i["text"], i["labels"]))

    # Build predicted spans using the prediction dataset and the gold dataset
    all_y_pred = build_prediction_spans(
        pred_dataset, gold_dataset, begin_instruction_tag, end_instruction_tag
    )

    #### Level 2
    f1_level_2 = 0
    precision_level_2 = 0
    recall_level_2 = 0
    label_f1_level_2 = 0
    label_precision_level_2 = 0
    label_recall_level_2 = 0
    try:
        for y_pred, y_true in zip(all_y_pred, all_y_true):
            p, r, f1 = text_label_only_p_r_f1(y_pred, y_true, NbClasses.LVL_2)
            label_precision_level_2 += p
            label_recall_level_2 += r
            label_f1_level_2 += f1 if not math.isnan(f1) else 0
            p, r, f1 = text_full_task_p_r_f1(y_pred, y_true)
            precision_level_2 += p
            recall_level_2 += r
            f1_level_2 += f1 if not math.isnan(f1) else 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    label_precision_level_2 /= len(all_y_pred)
    label_recall_level_2 /= len(all_y_pred)
    label_f1_level_2 /= len(all_y_pred)
    precision_level_2 /= len(all_y_pred)
    recall_level_2 /= len(all_y_pred)
    f1_level_2 /= len(all_y_pred)

    #### Level 1
    f1_level_1 = 0
    precision_level_1 = 0
    recall_level_1 = 0
    label_f1_level_1 = 0
    label_precision_level_1 = 0
    label_recall_level_1 = 0
    try:
        # all_y_pred = concatenate_sentences_to_spans_levels(deepcopy(all_y_pred_not_concatenated), level=1)
        for y_pred, y_true in zip(deepcopy(all_y_pred), all_y_true):
            # Convert labels from Level 2 to Level 1 for prediction spans
            for i in range(len(y_pred.spans)):
                y_pred.spans[i].label = LEVEL_2_TO_1[y_pred.spans[i].label]

            # Convert labels from Level 2 to Level 1 for ground truth spans
            for j in range(len(y_true.spans)):
                tmp_set_labels = set()
                for label in y_true.spans[j].labels:
                    if label is not None:
                        tmp_set_labels.add(LEVEL_2_TO_1[label])
                    else:
                        tmp_set_labels.add(None)
                y_true.spans[j].labels = tmp_set_labels

            # print(y_pred, y_true)
            p, r, f1 = text_label_only_p_r_f1(y_pred, y_true, NbClasses.LVL_1)
            label_precision_level_1 += p
            label_recall_level_1 += r
            label_f1_level_1 += f1 if not math.isnan(f1) else 0
            p, r, f1 = text_full_task_p_r_f1(y_pred, y_true)
            precision_level_1 += p
            recall_level_1 += r
            f1_level_1 += f1 if not math.isnan(f1) else 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    label_precision_level_1 /= len(all_y_pred)
    label_recall_level_1 /= len(all_y_pred)
    label_f1_level_1 /= len(all_y_pred)
    precision_level_1 /= len(all_y_pred)
    recall_level_1 /= len(all_y_pred)
    f1_level_1 /= len(all_y_pred)

    #### Level 0
    f1_level_0 = 0
    precision_level_0 = 0
    recall_level_0 = 0
    label_f1_level_0 = 0
    label_precision_level_0 = 0
    label_recall_level_0 = 0
    try:
        # all_y_pred = concatenate_sentences_to_spans_levels(deepcopy(all_y_pred_not_concatenated), level=0)
        for y_pred, y_true in zip(deepcopy(all_y_pred), all_y_true):
            for i in range(len(y_pred.spans)):
                if y_pred.spans[i].label == 0:
                    y_pred.spans[i].label = 0
                elif y_pred.spans[i].label >= 24:
                    y_pred.spans[i].label = 0
                else:
                    y_pred.spans[i].label = 1

            for j in range(len(y_true.spans)):
                tmp_set_labels = set()
                for label in y_true.spans[j].labels:
                    if label == 0 or label is None:
                        tmp_set_labels.add(0)
                    else:
                        tmp_set_labels.add(1)
                y_true.spans[j].labels = tmp_set_labels

            # print(y_pred, y_true)
            p, r, f1 = text_label_only_p_r_f1(y_pred, y_true, NbClasses.LVL_0)
            label_precision_level_0 += p
            label_recall_level_0 += r
            label_f1_level_0 += f1 if not math.isnan(f1) else 0
            p, r, f1 = text_full_task_p_r_f1(y_pred, y_true)
            precision_level_0 += p
            recall_level_0 += r
            f1_level_0 += f1 if not math.isnan(f1) else 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    label_precision_level_0 /= len(all_y_pred)
    label_recall_level_0 /= len(all_y_pred)
    label_f1_level_0 /= len(all_y_pred)
    precision_level_0 /= len(all_y_pred)
    recall_level_0 /= len(all_y_pred)
    f1_level_0 /= len(all_y_pred)

    return (
        precision_level_0,
        recall_level_0,
        f1_level_0,
        precision_level_1,
        recall_level_1,
        f1_level_1,
        precision_level_2,
        recall_level_2,
        f1_level_2,
        label_precision_level_0,
        label_recall_level_0,
        label_f1_level_0,
        label_precision_level_1,
        label_recall_level_1,
        label_f1_level_1,
        label_precision_level_2,
        label_recall_level_2,
        label_f1_level_2,
    )


def eval_dataset(dataset_path: str, results_path: str):
    with open("result_subjective_metric.csv", "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(
            [
                "Model",
                "Precision Level 0",
                "Recall Level 0",
                "F1 Level 0",
                "Precision Level 1",
                "Recall Level 1",
                "F1 Level 1",
                "Precision Level 2",
                "Recall Level 2",
                "F1 Level 2",
                "Label Precision Level 0",
                "Label Recall Level 0",
                "Label F1 Level 0",
                "Label Precision Level 1",
                "Label Recall Level 1",
                "Label F1 Level 1",
                "Label Precision Level 2",
                "Label Recall Level 2",
                "Label F1 Level 2",
            ]
        )
        for f_result in sorted(list(os.listdir(results_path))):
            print(f_result)
            if f_result.endswith(".jsonl"):
                model_name = f_result.replace("_level_2_results.jsonl", "")
                (
                    precision_level_0,
                    recall_level_0,
                    f1_level_0,
                    precision_level_1,
                    recall_level_1,
                    f1_level_1,
                    precision_level_2,
                    recall_level_2,
                    f1_level_2,
                    label_precision_level_0,
                    label_recall_level_0,
                    label_f1_level_0,
                    label_precision_level_1,
                    label_recall_level_1,
                    label_f1_level_1,
                    label_precision_level_2,
                    label_recall_level_2,
                    label_f1_level_2,
                ) = evaluate(dataset_path, os.path.join(results_path, f_result))
                csv_out.writerow(
                    [
                        model_name,
                        precision_level_0,
                        recall_level_0,
                        f1_level_0,
                        precision_level_1,
                        recall_level_1,
                        f1_level_1,
                        precision_level_2,
                        recall_level_2,
                        f1_level_2,
                        label_precision_level_0,
                        label_recall_level_0,
                        label_f1_level_0,
                        label_precision_level_1,
                        label_recall_level_1,
                        label_f1_level_1,
                        label_precision_level_2,
                        label_recall_level_2,
                        label_f1_level_2,
                    ]
                )


def evaluate_level_1(dataset_path: str, prediction_path: str):
    all_y_true = []

    gold_dataset = read_jsonl(dataset_path)
    pred_dataset = read_jsonl(prediction_path)

    begin_instruction_tag = ""
    end_instruction_tag = ""

    for model in MODELS_INSTUCTIONS_TAGS:
        if model in prediction_path:
            begin_instruction_tag = MODELS_INSTUCTIONS_TAGS[model][0]
            end_instruction_tag = MODELS_INSTUCTIONS_TAGS[model][1]
            break

        # Build ground truth spans for each instance in the gold dataset
    for i in gold_dataset:
        all_y_true.append(build_ground_truth_spans(i["text"], i["labels"]))

        # Build predicted spans using the prediction dataset and the gold dataset
    all_y_pred = build_prediction_spans(
        pred_dataset, gold_dataset, begin_instruction_tag, end_instruction_tag, level=1
    )

    #### Level 1
    f1_level_1 = 0
    precision_level_1 = 0
    recall_level_1 = 0
    label_f1_level_1 = 0
    label_precision_level_1 = 0
    label_recall_level_1 = 0
    try:
        # all_y_pred = concatenate_sentences_to_spans_levels(deepcopy(all_y_pred_not_concatenated), level=1)
        for y_pred, y_true in zip(all_y_pred, all_y_true):
            # Convert labels from Level 2 to Level 1 for ground truth spans
            for j in range(len(y_true.spans)):
                tmp_set_labels = set()
                for label in y_true.spans[j].labels:
                    if label is not None:
                        tmp_set_labels.add(LEVEL_2_TO_1[label])
                    else:
                        tmp_set_labels.add(None)
                y_true.spans[j].labels = tmp_set_labels

                # print(y_pred, y_true)
            p, r, f1 = text_label_only_p_r_f1(y_pred, y_true, NbClasses.LVL_1)
            label_precision_level_1 += p
            label_recall_level_1 += r
            label_f1_level_1 += f1 if not math.isnan(f1) else 0
            p, r, f1 = text_full_task_p_r_f1(y_pred, y_true)
            precision_level_1 += p
            recall_level_1 += r
            f1_level_1 += f1 if not math.isnan(f1) else 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    label_precision_level_1 /= len(all_y_pred)
    label_recall_level_1 /= len(all_y_pred)
    label_f1_level_1 /= len(all_y_pred)
    precision_level_1 /= len(all_y_pred)
    recall_level_1 /= len(all_y_pred)
    f1_level_1 /= len(all_y_pred)

    return (
        precision_level_1,
        recall_level_1,
        f1_level_1,
        label_precision_level_1,
        label_recall_level_1,
        label_f1_level_1,
    )
