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


def precision_score(
    pred_annotations: AnnotatedText, gold_annotations: AnnotatedText
) -> float:
    if len(pred_annotations) == 0:
        return 1

    sum_score = 0
    for pred in pred_annotations.spans:
        max_partial_score = 0
        for gold in gold_annotations.spans:
            if label_score(pred, gold) == 1:
                max_partial_score = max(
                    max_partial_score,
                    partial_overlap_score(pred, gold, PartialScoreType.PRED_SIZE),
                )
        sum_score += max_partial_score

    return sum_score / len(pred_annotations)


def recall_score(
    pred_annotations: AnnotatedText, gold_annotations: AnnotatedText
) -> float:
    sum_score = 0
    cnt_gold_spans = 0

    for gold in gold_annotations.spans:
        max_partial_score = 0
        if NOTHING_LABEL not in gold.labels:
            cnt_gold_spans += 1
            for pred in pred_annotations.spans:
                if label_score(pred, gold) == 1:
                    max_partial_score = max(
                        max_partial_score,
                        partial_overlap_score(pred, gold, PartialScoreType.GOLD_SIZE),
                    )
            sum_score += max_partial_score

    if cnt_gold_spans == 0:
        return 1

    return sum_score / cnt_gold_spans


def f1_score(pred_annotations: AnnotatedText, gold_annotations: AnnotatedText) -> float:
    precision = precision_score(pred_annotations, gold_annotations)
    recall = recall_score(pred_annotations, gold_annotations)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def precision_recall_f1_score(
    pred_annotations: AnnotatedText, gold_annotations: AnnotatedText
) -> Tuple[float, float, float]:
    precision = precision_score(pred_annotations, gold_annotations)
    recall = recall_score(pred_annotations, gold_annotations)
    f1 = f1_score(pred_annotations, gold_annotations)
    return precision, recall, f1


if __name__ == "__main__":
    text1 = "Lorem ipsum dolor sit amet"
    int1 = [0, 24]
    text2 = "Ut enim ad minim veniam"
    int2 = [25, 50]
    text3 = "Blabla et blabla et blabla"
    int3 = [51, 76]

    gd0 = AnnotatedText(
        [
            GroundTruthSpan(text1, {0, 1}, int1),
            GroundTruthSpan(text2, {2}, int2),
            GroundTruthSpan(text3, {3}, int3),
        ]
    )

    pd01 = AnnotatedText(
        [
            PredictionSpan(text2, 2, int2),
        ]
    )

    pd02 = AnnotatedText(
        [
            PredictionSpan(text1, 1, int1),
            PredictionSpan(text2, 2, int2),
        ]
    )

    # Case 0
    assert precision_score(pd01, gd0) == 1
    assert recall_score(pd01, gd0) == 1 / 2
    assert f1_score(pd01, gd0) == 2 / 3

    assert precision_score(pd02, gd0) == 1
    assert recall_score(pd02, gd0) == 1 / 2
    assert f1_score(pd02, gd0) == 2 / 3

    # Case 1
    gd1 = AnnotatedText([GroundTruthSpan(text1, {1, 0}, int1)])

    pd11 = AnnotatedText([PredictionSpan(text1, 1, int1)])

    pd12 = AnnotatedText([PredictionSpan(text1, 3, int1)])

    pd13 = AnnotatedText([])

    pd14 = AnnotatedText([PredictionSpan(text2, 1, int2)])

    pd15 = AnnotatedText([PredictionSpan(f"{text1} {text2}", 3, [0, 50])])

    assert precision_score(pd11, gd1) == 1
    assert recall_score(pd11, gd1) == 1
    assert f1_score(pd11, gd1) == 1

    assert precision_score(pd12, gd1) == 0
    assert recall_score(pd12, gd1) == 1
    assert f1_score(pd12, gd1) == 0

    assert precision_score(pd13, gd1) == 1
    assert recall_score(pd13, gd1) == 1
    assert f1_score(pd13, gd1) == 1

    assert precision_score(pd14, gd1) == 0
    assert recall_score(pd14, gd1) == 1
    assert f1_score(pd14, gd1) == 0

    assert precision_score(pd15, gd1) == 0
    assert recall_score(pd15, gd1) == 1
    assert f1_score(pd15, gd1) == 0

    # Case 1pr
    gd1pr = AnnotatedText([GroundTruthSpan(text1, {1}, int1)])

    pd1pr1 = AnnotatedText([PredictionSpan(text1, 1, int1)])

    pd1pr2 = AnnotatedText([PredictionSpan(text1, 3, int1)])

    pd1pr3 = AnnotatedText([PredictionSpan(text2, 1, int2)])

    pd1pr4 = AnnotatedText([])

    assert precision_score(pd1pr1, gd1pr) == 1
    assert recall_score(pd1pr1, gd1pr) == 1
    assert f1_score(pd1pr1, gd1pr) == 1

    assert precision_score(pd1pr2, gd1pr) == 0
    assert recall_score(pd1pr2, gd1pr) == 0
    assert f1_score(pd1pr2, gd1pr) == 0

    assert precision_score(pd1pr3, gd1pr) == 0
    assert recall_score(pd1pr3, gd1pr) == 0
    assert f1_score(pd1pr3, gd1pr) == 0

    assert precision_score(pd1pr4, gd1pr) == 1
    assert recall_score(pd1pr4, gd1pr) == 0
    assert f1_score(pd1pr4, gd1pr) == 0

    # Case 1dpr
    gd1dpr = AnnotatedText([])

    pd1dpr1 = AnnotatedText([PredictionSpan(text1, 1, int1)])
    pd1dpr2 = AnnotatedText([])

    assert precision_score(pd1dpr1, gd1dpr) == 0
    assert recall_score(pd1dpr1, gd1dpr) == 1
    assert f1_score(pd1dpr1, gd1dpr) == 0

    assert precision_score(pd1dpr2, gd1dpr) == 1
    assert recall_score(pd1dpr2, gd1dpr) == 1
    assert f1_score(pd1dpr2, gd1dpr) == 1

    # Case 2
    gd2 = AnnotatedText(
        [GroundTruthSpan(text1, {1, 0}, int1), GroundTruthSpan(text2, {2}, int2)]
    )

    pd21 = AnnotatedText([PredictionSpan(text1, 1, int1)])

    pd22 = AnnotatedText([PredictionSpan(text1, 3, int1)])

    pd23 = AnnotatedText([PredictionSpan(text2, 2, int2)])

    pd24 = AnnotatedText([PredictionSpan(text2, 3, int2)])

    assert precision_score(pd21, gd2) == 1
    assert recall_score(pd21, gd2) == 0
    assert f1_score(pd21, gd2) == 0

    assert precision_score(pd22, gd2) == 0
    assert recall_score(pd22, gd2) == 0
    assert f1_score(pd22, gd2) == 0

    assert precision_score(pd23, gd2) == 1
    assert recall_score(pd23, gd2) == 1
    assert f1_score(pd23, gd2) == 1

    assert precision_score(pd24, gd2) == 0
    assert recall_score(pd24, gd2) == 0
    assert f1_score(pd24, gd2) == 0

    # Case 3
    gd3 = AnnotatedText([GroundTruthSpan(f"{text1} {text2}", {1, 0}, [0, 50])])

    pd31 = AnnotatedText([PredictionSpan(f"{text1} {text2}", 1, [0, 50])])

    pd32 = AnnotatedText([])

    pd33 = AnnotatedText([PredictionSpan(text1, 1, int1)])

    pd34 = AnnotatedText([PredictionSpan(f"{text1} {text2}", 3, [0, 50])])

    pd35 = AnnotatedText([PredictionSpan(text2, 3, int2)])

    assert precision_score(pd31, gd3) == 1
    assert recall_score(pd31, gd3) == 1
    assert f1_score(pd31, gd3) == 1

    assert precision_score(pd32, gd3) == 1
    assert recall_score(pd32, gd3) == 1
    assert f1_score(pd32, gd3) == 1

    assert precision_score(pd33, gd3) == 1
    assert recall_score(pd33, gd3) == 1
    assert f1_score(pd33, gd3) == 1

    assert precision_score(pd34, gd3) == 0
    assert recall_score(pd34, gd3) == 1
    assert f1_score(pd34, gd3) == 0

    assert precision_score(pd35, gd3) == 0
    assert recall_score(pd35, gd3) == 1
    assert f1_score(pd35, gd3) == 0
