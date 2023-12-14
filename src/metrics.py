from enum import Enum
from itertools import chain, combinations
from typing import Any, List, Set, Tuple, Union

NOTHING_LABEL = 0


class NbClasses(Enum):
    LVL_0 = 1
    LVL_1 = 3
    LVL_2 = 23


class Span:
    def __init__(self, span: str):
        self.span = span

    def __str__(self) -> str:
        return f"{self.span}"

    def __repr__(self) -> str:
        return self.__str__()


class PredictionSpan(Span):
    def __init__(self, span: str, label: Union[int, None], interval: List[int]):
        super().__init__(span)
        self.label = label
        self.interval = interval

    def __eq__(self, other):
        if not isinstance(other, PredictionSpan):
            return False
        return self.span == other.span

    def __str__(self) -> str:
        return super().__str__() + f" - {self.interval} - {self.label}"

    def __repr__(self) -> str:
        return self.__str__()


class GroundTruthSpan(Span):
    def __init__(self, span: str, labels: Set[Union[int, None]], interval: List[int]):
        super().__init__(span)
        self.labels = labels
        self.interval = interval

    def __eq__(self, other):
        if not isinstance(other, GroundTruthSpan):
            return False
        return (
            self.span == other.span
            and self.labels == other.labels
            and self.interval == other.interval
        )

    def __str__(self) -> str:
        return super().__str__() + f" - {self.interval} - {self.labels}"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(
            (self.span, tuple(sorted(self.labels)), self.interval[0], self.interval[1])
        )


class ScoreType(Enum):
    PRECISION = 1
    RECALL = 2


class AnnotatedText:
    def __init__(self, spans: List[Union[PredictionSpan, GroundTruthSpan]]):
        self.spans = spans

    def __len__(self):
        return len(self.spans)

    def __str__(self) -> str:
        return f"{self.spans}"

    def __repr__(self) -> str:
        return self.__str__()

    def generate_label_conjunctions(
        self, score_type: ScoreType, to_numeric_labels=False
    ) -> List[List[Any]]:
        """Return a list of all possible conjunctions of spans in the text.
        If to_numeric_labels is True, then we keep only the numeric labels of the spans (removing none and nothing values).
        Otherwise, returns GroundTruthSpan with one label only.
        score_type: PRECISION or RECALL because we don't handle the disjunctions the same way (OR for precision, XOR for recall)
        """
        # Validate and process spans
        label_sets = [
            s
            for s in self.spans
            if isinstance(s, GroundTruthSpan) and s.labels and s.labels != {None}
        ]
        if not all(isinstance(span, GroundTruthSpan) for span in self.spans):
            raise Exception("This method is only useful for gold spans.")

        def generate_ground_truths(labels, span):
            return [
                GroundTruthSpan(span.span, {label}, span.interval) for label in labels
            ]

        def combine_label_sets(label_sets):
            if not label_sets:
                return [[]]

            first_set, rest_sets = label_sets[0], label_sets[1:]
            combinations_of_rest = combine_label_sets(rest_sets)

            if score_type == ScoreType.PRECISION:
                return [
                    list(ground_truths) + list(combination)
                    for combination in combinations_of_rest
                    for ground_truths in chain.from_iterable(
                        combinations(
                            generate_ground_truths(list(first_set.labels), first_set), r
                        )
                        for r in range(1, len(first_set.labels) + 1)
                    )
                ]
            else:
                return [
                    generate_ground_truths([label], first_set) + combination
                    for label in first_set.labels
                    for combination in combinations_of_rest
                ]

        all_combinations = combine_label_sets(label_sets)

        # Convert combinations to the desired format
        def process_combination(combination):
            if to_numeric_labels:
                return list(
                    {
                        label
                        for span in combination
                        for label in span.labels
                        if isinstance(label, int) and label > NOTHING_LABEL
                    }
                )
            return [
                span
                for span in combination
                if len(span.labels) == 1 and NOTHING_LABEL not in span.labels
            ]

        result = [process_combination(combination) for combination in all_combinations]

        def are_lists_equal(list1, list2):
            # Step 1: Check lengths
            if len(list1) != len(list2):
                return False

            # Step 2: Iterate through elements
            for item1, item2 in zip(list1, list2):
                # Step 3: Check for equality (customize this part as needed)
                if item1 != item2:
                    return False

            # Step 4: All elements are equal
            return True

        result = [
            lst
            for i, lst in enumerate(result)
            if not any(
                are_lists_equal(lst, lst2) for j, lst2 in enumerate(result) if j < i
            )
        ]
        return result


def label_score(pred: PredictionSpan, gold: GroundTruthSpan) -> float:
    """Return 1 if pred values are in gold, 0 else.
    Correspond to delta in the C function in the paper"""
    # equivalent to δ(a, b) or F ⊆ F′
    # can be change if necessary, to take into account close labels for instance
    if pred.label in gold.labels:
        return 1
    return 0


class PartialScoreType(Enum):
    JACCARD_INDEX = 1
    PRED_SIZE = 2
    GOLD_SIZE = 3


def partial_overlap_score(
    pred: PredictionSpan,
    gold: GroundTruthSpan,
    partial_score_type: PartialScoreType = PartialScoreType.JACCARD_INDEX,
) -> float:
    """Return the jaccard index between two spans, or use FINE Grained analysis score.
    Corresponds to the fraction in the C function in the paper"""
    # equivalent to Jaccard(a, b) or F ∩ F′ / F ∪ F′
    a, b = pred.interval
    c, d = gold.interval
    # Compute the intersection
    intersection_start = max(a, c)
    intersection_end = min(b, d)
    intersection_length = max(0, intersection_end - intersection_start)

    if partial_score_type == PartialScoreType.JACCARD_INDEX:
        # Compute the union
        union_length = (b - a) + (d - c) - intersection_length

        # Check for no overlap
        if union_length == 0:
            return 0

        # Compute the Jaccard index
        return intersection_length / union_length
    elif partial_score_type == PartialScoreType.PRED_SIZE:
        return intersection_length / (b - a) if (b - a) else 0
    elif partial_score_type == PartialScoreType.GOLD_SIZE:
        return intersection_length / (d - c) if (d - c) else 0


def text_full_task_precision(
    pred_corpus: AnnotatedText, gold_corpus: AnnotatedText
) -> float:
    """Return the precision score of a prediction (spans + labels)."""
    if not pred_corpus.spans and not gold_corpus.spans:
        # there was nothing to predict and nothing was found
        return 1
    disjunct = gold_corpus.generate_label_conjunctions(ScoreType.PRECISION)
    if disjunct == [[]]:
        # there was nothing to predict
        if any(s.label != NOTHING_LABEL for s in pred_corpus.spans):
            # there was something predicted
            return 0
        # there was nothing predicted
        return 1
    precision_scores = []
    for y_trues in disjunct:
        p_sum = []
        for pred_span in pred_corpus.spans:
            p_sum += [0]
            for gold_span in y_trues:
                ls = label_score(pred_span, gold_span)
                if ls == 0:
                    continue
                p_pos = partial_overlap_score(
                    pred_span, gold_span, PartialScoreType.PRED_SIZE
                )
                if p_pos * ls > p_sum[-1]:
                    p_sum[-1] = p_pos * ls
        # print(p_sum)
        p_sum = sum(p_sum)

        p_denominator = len([s for s in pred_corpus.spans if s.label != NOTHING_LABEL])
        if p_denominator == 0:
            if len(y_trues) == 0:
                precision_scores.append(1)
            elif len(y_trues) > 0:
                precision_scores.append(0)
            else:
                raise Exception(
                    "This should not happen: denominator = 0 but len(y_true) > 0"
                )
        else:
            precision_score = p_sum / p_denominator
            if precision_score > 1:
                raise Exception("This should not happen: precision > 1")
            precision_scores.append(precision_score)
    precision = max(precision_scores)
    return precision


def text_full_task_recall(
    pred_corpus: AnnotatedText, gold_corpus: AnnotatedText
) -> float:
    """Return the recall score of a prediction (spans + labels)."""
    if not pred_corpus.spans and not gold_corpus.spans:
        # there was nothing to predict and nothing was found
        return 1
    disjunct = gold_corpus.generate_label_conjunctions(ScoreType.RECALL)
    if disjunct == [[]]:
        # there was nothing to predict
        if any(s.label != NOTHING_LABEL for s in pred_corpus.spans):
            # there was something predicted
            return 0
        # there was nothing predicted
        return 1
    recall_scores = []
    for y_trues in disjunct:
        r_sum = []
        for gold_span in y_trues:
            r_sum += [0]
            for pred_span in pred_corpus.spans:
                ls = label_score(pred_span, gold_span)
                if ls == 0:
                    continue
                r_pos = partial_overlap_score(
                    pred_span, gold_span, PartialScoreType.GOLD_SIZE
                )
                if r_pos * ls > r_sum[-1]:
                    r_sum[-1] = r_pos * ls
        r_sum = sum(r_sum)

        # we don't count spans wo fallacies:
        r_denominator = len(
            [s for s in y_trues if None not in s.labels and 0 not in s.labels]
        )
        if r_denominator == 0:
            if len(pred_corpus) == 0:
                recall_scores.append(1)
            elif len(pred_corpus) > 0:
                recall_scores.append(0)
            else:
                raise Exception(
                    "This should not happen: divide by 0 on recall denominator"
                )
        else:
            recall_score = r_sum / r_denominator
            recall_scores.append(recall_score)
    recall = max(recall_scores)
    return recall


def text_full_task_p_r_f1(
    pred_corpus: AnnotatedText, gold_corpus: AnnotatedText
) -> Tuple[float, float, float]:
    """Return the precision, recall, and F1 scores of a prediction (spans + labels)."""
    if not pred_corpus.spans and not gold_corpus.spans:
        # there was nothing to predict and nothing was found
        return 1, 1, 1

    precision = text_full_task_precision(pred_corpus, gold_corpus)
    recall = text_full_task_recall(pred_corpus, gold_corpus)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def text_label_only_precision(
    pred_text: AnnotatedText, gold_text: AnnotatedText, nb_classes: NbClasses
) -> float:
    """Compute the precision score for labels only."""
    if not pred_text.spans and not gold_text.spans:
        # there was nothing to predict and nothing was found
        return 1

    y_pred = {
        s.label
        for s in pred_text.spans
        if isinstance(s.label, int)
        and s.label != NOTHING_LABEL
        and s.label - 1 < nb_classes.value
    }
    y_trues = [
        {
            list(s.labels)[0]
            for s in y_true
            if isinstance(list(s.labels)[0], int)
            and list(s.labels)[0] != NOTHING_LABEL
            and list(s.labels)[0] - 1 < nb_classes.value
        }
        for y_true in gold_text.generate_label_conjunctions(ScoreType.PRECISION)
    ]
    precision_results = []
    for y_true in y_trues:
        tp = len(set(y_pred).intersection(y_true))
        if tp == 0:
            if len(y_pred) == 0 and len(y_true) == 0:
                precision_results.append(1)
            else:
                precision_results.append(0)
            continue
        fp = len(set(y_pred).difference(y_true))
        precision_score = tp / (tp + fp)
        precision_results.append(precision_score)
    precision = max(precision_results)
    return precision


def text_label_only_recall(
    pred_text: AnnotatedText, gold_text: AnnotatedText, nb_classes: NbClasses
) -> float:
    """Compute the recall score for labels only."""
    if not pred_text.spans and not gold_text.spans:
        # there was nothing to predict and nothing was found
        return 1

    y_pred = {
        s.label
        for s in pred_text.spans
        if isinstance(s.label, int)
        and s.label != NOTHING_LABEL
        and s.label - 1 < nb_classes.value
    }
    y_trues = [
        {
            list(s.labels)[0]
            for s in y_true
            if isinstance(list(s.labels)[0], int)
            and list(s.labels)[0] != NOTHING_LABEL
            and list(s.labels)[0] - 1 < nb_classes.value
        }
        for y_true in gold_text.generate_label_conjunctions(ScoreType.RECALL)
    ]
    recall_results = []
    for y_true in y_trues:
        tp = len(set(y_pred).intersection(y_true))
        if tp == 0:
            if len(y_pred) == 0 and len(y_true) == 0:
                recall_results.append(1)
            else:
                recall_results.append(0)
            continue
        fn = len(set(y_true).difference(y_pred))
        recall_score = tp / (tp + fn)
        recall_results.append(recall_score)
    recall = max(recall_results)
    return recall


def text_label_only_p_r_f1(
    pred_text: AnnotatedText, gold_text: AnnotatedText, nb_classes: NbClasses
) -> Tuple[float, float, float]:
    """Compute the precision, recall, and F1 scores for labels only."""
    if not pred_text.spans and not gold_text.spans:
        # there was nothing to predict and nothing was found
        return 1, 1, 1

    precision = text_label_only_precision(pred_text, gold_text, nb_classes)
    recall = text_label_only_recall(pred_text, gold_text, nb_classes)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


if __name__ == "__main__":
    # Example 1: Simple case with single labels in each span
    spans_example1 = [
        GroundTruthSpan("span1", {1}, [0, 1]),
        GroundTruthSpan("span2", {2}, [1, 2]),
    ]
    corpus_example1 = AnnotatedText(spans_example1)

    # Example 2: Case with multiple labels in some spans
    spans_example2 = [
        GroundTruthSpan("span1", {1, 2}, [0, 1]),
        GroundTruthSpan("span2", {3}, [1, 2]),
        GroundTruthSpan("span3", {4, 5}, [2, 3]),
    ]
    corpus_example2 = AnnotatedText(spans_example2)

    # Example 3: Case with a None value in one of the spans
    spans_example3 = [
        GroundTruthSpan("span1", {1, None}, [0, 1]),
        GroundTruthSpan("span2", {2, 3}, [1, 2]),
    ]
    corpus_example3 = AnnotatedText(spans_example3)

    # Example 4: Case with an empty label set in one of the spans
    spans_example4 = [
        GroundTruthSpan("span1", {1, 2}, [0, 1]),
        GroundTruthSpan("span2", set(), [1, 2]),  # Empty label set
        GroundTruthSpan("span3", {3}, [2, 3]),
    ]
    corpus_example4 = AnnotatedText(spans_example4)

    # Example 5:
    spans_example5 = [
        GroundTruthSpan("span1", {1, None}, [0, 1]),
    ]
    corpus_example5 = AnnotatedText(spans_example5)

    # Example 6:
    spans_example6 = [
        GroundTruthSpan("span1", {1, None}, [0, 1]),
        GroundTruthSpan("span1", {3}, [0, 1]),
    ]
    corpus_example6 = AnnotatedText(spans_example6)

    # Run the function for each example
    result1 = corpus_example1.generate_label_conjunctions(True)
    result2 = corpus_example2.generate_label_conjunctions(True)
    result3 = corpus_example3.generate_label_conjunctions(True)
    result4 = corpus_example4.generate_label_conjunctions(True)
    result5 = corpus_example5.generate_label_conjunctions(True)
    result6 = corpus_example6.generate_label_conjunctions(True)

    assert result1 == [[1, 2]]
    assert result2 == [[1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5]]
    assert result3 == [[1, 2], [1, 3], [2], [3]]
    assert result4 == [[1, 3], [2, 3]]
    assert result5 == [[1], []]
    assert result6 == [[1, 3], [3]]

    ##########

    #### Pred: a and b and c, Gold: a and (b or c)
    pred_spans_correct = [
        PredictionSpan("Text1", 1, [0, 10]),
        PredictionSpan("Text1", 2, [0, 10]),
        PredictionSpan("Text1", 3, [0, 10]),
    ]
    pred_text_correct = AnnotatedText(pred_spans_correct)

    # Gold Spans - Same as Prediction
    gold_spans_same = [
        GroundTruthSpan("Text1", {1}, [0, 10]),
        GroundTruthSpan("Text1", {2, 3}, [0, 10]),
    ]
    gold_text_same = AnnotatedText(gold_spans_same)

    # Testing text_full_task_p_r_f1 - All Correct
    p, r, f1 = text_full_task_p_r_f1(pred_text_correct, gold_text_same)
    assert p == 1, f"Expected precision is 1, but got {p} (All Correct)"
    assert r == 1, f"Expected recall is 1, but got {r} (All Correct)"
    assert f1 == 1, f"Expected F1 score is 1, but got {f1} (All Correct)"

    # Testing text_label_only_p_r_f1 - All Correct
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_correct, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 1, f"Expected precision is 1, but got {p_l} (All Correct)"
    assert r_l == 1, f"Expected recall is 1, but got {r_l} (All Correct)"
    assert f1_l == 1, f"Expected F1 score is 1, but got {f1_l} (All Correct)"

    #### Prediction Spans - All Correct
    pred_spans_correct = [
        PredictionSpan("Text1", 1, [0, 10]),
        PredictionSpan("Text2", 2, [11, 20]),
    ]
    pred_text_correct = AnnotatedText(pred_spans_correct)

    # Gold Spans - Same as Prediction
    gold_spans_same = [
        GroundTruthSpan("Text1", {1}, [0, 10]),
        GroundTruthSpan("Text2", {2}, [11, 20]),
    ]
    gold_text_same = AnnotatedText(gold_spans_same)

    # Testing text_full_task_p_r_f1 - All Correct
    p, r, f1 = text_full_task_p_r_f1(pred_text_correct, gold_text_same)
    assert p == 1, f"Expected precision is 1, but got {p} (All Correct)"
    assert r == 1, f"Expected recall is 1, but got {r} (All Correct)"
    assert f1 == 1, f"Expected F1 score is 1, but got {f1} (All Correct)"

    # Testing text_label_only_p_r_f1 - All Correct
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_correct, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 1, f"Expected precision is 1, but got {p_l} (All Correct)"
    assert r_l == 1, f"Expected recall is 1, but got {r_l} (All Correct)"
    assert f1_l == 1, f"Expected F1 score is 1, but got {f1_l} (All Correct)"

    #### Prediction Spans - All Incorrect
    pred_spans_incorrect = [
        PredictionSpan("Text1", 3, [200, 210]),  # Wrong label and spans
        PredictionSpan("Text2", 4, [311, 320]),  # Wrong labelands spans
    ]
    pred_text_incorrect = AnnotatedText(pred_spans_incorrect)

    # Testing text_full_task_p_r_f1 - All Incorrect
    p, r, f1 = text_full_task_p_r_f1(pred_text_incorrect, gold_text_same)
    assert p == 0, f"Expected precision is 0, but got {p} (All Incorrect)"
    assert r == 0, f"Expected recall is 0, but got {r} (All Incorrect)"
    assert f1 == 0, f"Expected F1 score is 0, but got {f1} (All Incorrect)"

    # Testing text_label_only_p_r_f1 - All Incorrect
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_incorrect, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 0, f"Expected precision is 0, but got {p_l} (All Incorrect)"
    assert r_l == 0, f"Expected recall is 0, but got {r_l} (All Incorrect)"
    assert f1_l == 0, f"Expected F1 score is 0, but got {f1_l} (All Incorrect)"

    #### Prediction Spans - same spans but different labels
    pred_spans_incorrect = [
        PredictionSpan("Text1", 3, [0, 10]),  # Wrong labns
        PredictionSpan("Text2", 4, [11, 20]),  # Wrong label
    ]
    pred_text_incorrect = AnnotatedText(pred_spans_incorrect)

    # Testing text_full_task_p_r_f1 - All Incorrect
    p, r, f1 = text_full_task_p_r_f1(pred_text_incorrect, gold_text_same)
    assert p == 0, f"Expected precision is 0, but got {p} (All Incorrect)"
    assert r == 0, f"Expected recall is 0, but got {r} (All Incorrect)"
    assert f1 == 0, f"Expected F1 score is 0, but got {f1} (All Incorrect)"

    # Testing text_label_only_p_r_f1 - All Incorrect
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_incorrect, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 0, f"Expected precision is 0, but got {p_l} (All Incorrect)"
    assert r_l == 0, f"Expected recall is 0, but got {r_l} (All Incorrect)"
    assert f1_l == 0, f"Expected F1 score is 0, but got {f1_l} (All Incorrect)"

    #### Prediction Spans - same labels, but none intersecting spans
    pred_spans_incorrect = [
        PredictionSpan("Text1", 1, [200, 210]),  # Wrong label and spans
        PredictionSpan("Text2", 2, [311, 320]),  # Wrong labelands spans
    ]
    pred_text_incorrect = AnnotatedText(pred_spans_incorrect)

    # Testing text_full_task_p_r_f1 - All Incorrect
    p, r, f1 = text_full_task_p_r_f1(pred_text_incorrect, gold_text_same)
    assert p == 0, f"Expected precision is 0, but got {p} (All Incorrect)"
    assert r == 0, f"Expected recall is 0, but got {r} (All Incorrect)"
    assert f1 == 0, f"Expected F1 score is 0, but got {f1} (All Incorrect)"

    # Testing text_label_only_p_r_f1 - All Incorrect
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_incorrect, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 1, f"Expected precision is 1, but got {p_l} (All Incorrect)"
    assert r_l == 1, f"Expected recall is 1, but got {r_l} (All Incorrect)"
    assert f1_l == 1, f"Expected F1 score is 1, but got {f1_l} (All Incorrect)"

    #### Prediction Spans - same labels, but some intersecting spans
    pred_spans_incorrect = [
        PredictionSpan("Text1", 1, [0, 5]),  # Wrong label and spans
        PredictionSpan("Text2", 2, [311, 320]),  # Wrong labelands spans
    ]
    pred_text_incorrect = AnnotatedText(pred_spans_incorrect)

    # Testing text_full_task_p_r_f1 - All Incorrect
    p, r, f1 = text_full_task_p_r_f1(pred_text_incorrect, gold_text_same)
    assert p == 0.5, f"Expected precision is 0.5, but got {p} (All Incorrect)"
    assert r == 0.25, f"Expected recall is 0.25, but got {r} (All Incorrect)"
    assert f1 == 1 / 3, f"Expected F1 score is 1/3, but got {f1} (All Incorrect)"

    # Testing text_label_only_p_r_f1 - All Incorrect
    p_l, r_l, f1_l = text_label_only_p_r_f1(
        pred_text_incorrect, gold_text_same, NbClasses.LVL_2
    )
    assert p_l == 1, f"Expected precision is 1, but got {p_l} (All Incorrect)"
    assert r_l == 1, f"Expected recall is 1, but got {r_l} (All Incorrect)"
    assert f1_l == 1, f"Expected F1 score is 1, but got {f1_l} (All Incorrect)"

    #### Mixed
    pred_spans = [
        PredictionSpan("Text1", 1, [0, 10]),  # Correct
        PredictionSpan("Text2", 1, [211, 220]),  # Incorrect label and spans
    ]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("Text1", {1}, [0, 10]),
        GroundTruthSpan("Text2", {2}, [11, 20]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 0.5, f"Expected precision is 0.5 but got {p}"
    assert r == 0.5, f"Expected recall is 0.5 but got {r}"
    assert f1 == 0.5, f"Expected F1 is 0.5 but got {f1}"

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"
    assert r_l == 0.5, f"Expected recall is 0.5 but got {r_l}"
    assert f1_l == 2 / 3, f"Expected F1 is 2/3 but got {f1_l}"

    #### No Predictions
    pred_spans = []  # Empty predictions
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("Text1", {1}, [0, 10]),
        GroundTruthSpan("Text2", {2}, [11, 20]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 0, f"Expected precision is 0 but got {p}"  # No false positives
    assert r == 0, f"Expected recall is 0 but got {r}"  # No true positives
    assert f1 == 0, f"Expected F1 is 0 but got {f1}"  # Precision and Recall are 0

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 0, f"Expected precision is 0 but got {p_l}"  # No false positives
    assert r_l == 0, f"Expected recall is 0 but got {r_l}"  # No true positives
    assert f1_l == 0, f"Expected F1 is 0 but got {f1_l}"  # Precision and Recall are 0

    #### Prediction Spans: No Gold Labels
    pred_spans = [
        PredictionSpan("Text1", 1, [0, 10]),
        PredictionSpan("Text2", 2, [11, 20]),
    ]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = []  # Empty gold labels
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 0, f"Expected precision is 0 but got {p}"  # No true positives
    assert r == 0, f"Expected recall is 0 but got {r}"  # No false negatives
    assert f1 == 0, f"Expected F1 is 0 but got {f1}"  # Precision is 0

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 0, f"Expected precision is 0 but got {p_l}"  # No true positives
    assert r_l == 0, f"Expected recall is 0 but got {r_l}"  # No false negatives
    assert f1_l == 0, f"Expected F1 is 0 but got {f1_l}"  # Precision is 0

    #### Prediction Spans with a label outside the class range
    pred_spans = [
        PredictionSpan("Text1", 25, [0, 10]),  # Label outside class range
        PredictionSpan("Text2", 2, [11, 20]),
    ]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("Text1", {1}, [0, 10]),
        GroundTruthSpan("Text2", {2}, [11, 20]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 0.5, f"Expected precision is 0.5 but got {p}"
    assert r == 0.5, f"Expected recall is 0.5 but got {r}"
    assert f1 == 0.5, f"Expected F1 is 0.5 but got {f1}"

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"
    assert r_l == 0.5, f"Expected recall is 0.5 but got {r_l}"
    assert f1_l == 2 / 3, f"Expected F1 is 2/3 but got {f1_l}"

    #### Prediction Spans: No Gold Labels and no pred
    pred_spans = []
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = []  # Empty gold labels
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 1, f"Expected precision is 1 but got {p}"  # No true positives
    assert r == 1, f"Expected recall is 1 but got {r}"  # No false negatives
    assert f1 == 1, f"Expected F1 is 1 but got {f1}"  # Precision is 0

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    #### 1 OR Nothing + predict something
    pred_spans = [PredictionSpan("span1", 1, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {1, None}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 1, f"Expected precision is 1 but got {p}"  # No true positives
    assert r == 1, f"Expected recall is 1 but got {r}"  # No false negatives
    assert f1 == 1, f"Expected F1 is 1 but got {f1}"  # Precision is 0

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    #### 1 OR Nothing + predict nothing
    pred_spans = []
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {1, None}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_full_task_p_r_f1
    p, r, f1 = text_full_task_p_r_f1(pred_text, gold_text)
    assert p == 1, f"Expected precision is 1 but got {p}"  # No true positives
    assert r == 1, f"Expected recall is 1 but got {r}"  # No false negatives
    assert f1 == 1, f"Expected F1 is 1 but got {f1}"  # Precision is 0

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_1)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # LEVEL 0
    # Both output 1:
    pred_spans = [PredictionSpan("span1", 1, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {1}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_0)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # Both output 0:
    pred_spans = [PredictionSpan("span1", 0, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {0}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_0)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # Model predict 1:
    pred_spans = [PredictionSpan("span1", 1, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {0}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_0)
    assert p_l == 0, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 0, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 0, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # Model predict 1:
    pred_spans = [PredictionSpan("span1", 1, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {0, 1}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_0)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # Model predict 1:
    pred_spans = [PredictionSpan("span1", 0, [0, 1])]
    pred_text = AnnotatedText(pred_spans)

    # Gold Spans
    gold_spans = [
        GroundTruthSpan("span1", {0, 1}, [0, 1]),
    ]
    gold_text = AnnotatedText(gold_spans)

    # Test for text_label_only_p_r_f1
    p_l, r_l, f1_l = text_label_only_p_r_f1(pred_text, gold_text, NbClasses.LVL_0)
    assert p_l == 1, f"Expected precision is 1 but got {p_l}"  # No true positives
    assert r_l == 1, f"Expected recall is 1 but got {r_l}"  # No false negatives
    assert f1_l == 1, f"Expected F1 is 1 but got {f1_l}"  # Precision is 0

    # Model predict 1:
    pred_spans = []
