import csv
import math
import os
import traceback
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from src.evaluate import LEVEL_2_NUMERIC, LEVEL_2_TO_1, build_ground_truth_spans
from src.metrics import AnnotatedText, NbClasses, PredictionSpan, text_label_only_p_r_f1
from src.new_metrics import text_full_task_p_r_f1
from src.utils import read_jsonl


def build_prediction_user_spans(text: str, labels: list[list[Any]]):
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
        dict_labels[i] = 0

    # Construct the list of GroundTruthSpan objects
    ground_truth_spans = []

    for idx in dict_labels:
        # Create a GroundTruthSpan for each labeled and uncovered span
        if dict_labels[idx] == 0:
            ground_truth_spans.append(
                PredictionSpan(
                    text[idx[0] : idx[1]], dict_labels[idx], [idx[0], idx[1]]
                )
            )
        else:
            for label in dict_labels[idx]:
                ground_truth_spans.append(
                    PredictionSpan(text[idx[0] : idx[1]], label, [idx[0], idx[1]])
                )

    return AnnotatedText(ground_truth_spans)


def user_evaluation(user_prediction_path: str):
    gold_dataset = read_jsonl("datasets/user_study_examples_with_labels.jsonl")
    pred_dataset = read_jsonl(user_prediction_path)

    all_y_true = []
    all_y_pred = []

    for gold, pred in zip(gold_dataset, pred_dataset):
        all_y_true.append(build_ground_truth_spans(gold["text"], gold["labels"]))
        all_y_pred.append(build_prediction_user_spans(pred["text"], pred["label"]))

    #### Level 2
    f1_level_2 = 0
    precision_level_2 = 0
    recall_level_2 = 0
    label_f1_level_2 = 0
    label_precision_level_2 = 0
    label_recall_level_2 = 0
    try:
        # all_y_pred = concatenate_sentences_to_spans_levels(all_y_pred_not_concatenated, level=2)
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


def users_study_evaluation():
    with open("users_study.csv", "w") as out:
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

        users_result_path = "datasets/users_results/"
        avg_precision_level_0 = 0
        avg_recall_level_0 = 0
        avg_f1_level_0 = 0
        avg_precision_level_1 = 0
        avg_recall_level_1 = 0
        avg_f1_level_1 = 0
        avg_precision_level_2 = 0
        avg_recall_level_2 = 0
        avg_f1_level_2 = 0
        avg_label_precision_level_0 = 0
        avg_label_recall_level_0 = 0
        avg_label_f1_level_0 = 0
        avg_label_precision_level_1 = 0
        avg_label_recall_level_1 = 0
        avg_label_f1_level_1 = 0
        avg_label_precision_level_2 = 0
        avg_label_recall_level_2 = 0
        avg_label_f1_level_2 = 0
        cnt = 0
        for user_result in sorted(list(os.listdir(users_result_path))):
            if user_result.endswith(".jsonl"):
                user_id = user_result.split(".")[0]
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
                ) = user_evaluation(os.path.join(users_result_path, user_result))
                csv_out.writerow(
                    [
                        user_id,
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
                avg_precision_level_0 += precision_level_0
                avg_recall_level_0 += recall_level_0
                avg_f1_level_0 += f1_level_0
                avg_precision_level_1 += precision_level_1
                avg_recall_level_1 += recall_level_1
                avg_f1_level_1 += f1_level_1
                avg_precision_level_2 += precision_level_2
                avg_recall_level_2 += recall_level_2
                avg_f1_level_2 += f1_level_2
                avg_label_precision_level_0 += label_precision_level_0
                avg_label_recall_level_0 += label_recall_level_0
                avg_label_f1_level_0 += label_f1_level_0
                avg_label_precision_level_1 += label_precision_level_1
                avg_label_recall_level_1 += label_recall_level_1
                avg_label_f1_level_1 += label_f1_level_1
                avg_label_precision_level_2 += label_precision_level_2
                avg_label_recall_level_2 += label_recall_level_2
                avg_label_f1_level_2 += label_f1_level_2
                cnt += 1

        avg_precision_level_0 /= cnt
        avg_recall_level_0 /= cnt
        avg_f1_level_0 /= cnt
        avg_precision_level_1 /= cnt
        avg_recall_level_1 /= cnt
        avg_f1_level_1 /= cnt
        avg_precision_level_2 /= cnt
        avg_recall_level_2 /= cnt
        avg_f1_level_2 /= cnt
        avg_label_precision_level_0 /= cnt
        avg_label_recall_level_0 /= cnt
        avg_label_f1_level_0 /= cnt
        avg_label_precision_level_1 /= cnt
        avg_label_recall_level_1 /= cnt
        avg_label_f1_level_1 /= cnt
        avg_label_precision_level_2 /= cnt
        avg_label_recall_level_2 /= cnt
        avg_label_f1_level_2 /= cnt

        csv_out.writerow(
            [
                "Average",
                avg_precision_level_0,
                avg_recall_level_0,
                avg_f1_level_0,
                avg_precision_level_1,
                avg_recall_level_1,
                avg_f1_level_1,
                avg_precision_level_2,
                avg_recall_level_2,
                avg_f1_level_2,
                avg_label_precision_level_0,
                avg_label_recall_level_0,
                avg_label_f1_level_0,
                avg_label_precision_level_1,
                avg_label_recall_level_1,
                avg_label_f1_level_1,
                avg_label_precision_level_2,
                avg_label_recall_level_2,
                avg_label_f1_level_2,
            ]
        )
