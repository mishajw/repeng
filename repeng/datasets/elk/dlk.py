from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from repeng.datasets.elk.types import BinaryRow, DlkDatasetId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle


@dataclass
class _DatasetSpec:
    name: str
    subset: str | None = None
    validation_name: str = "validation"


@dataclass
class _DlkTemplate:
    template: str
    labels: list[str]
    args: list[str]
    insert_label_options: bool = True


_DATASET_SPECS: dict[DlkDatasetId, _DatasetSpec] = {
    # Sentiment classification
    "imdb": _DatasetSpec("imdb", validation_name="test"),
    "amazon_polarity": _DatasetSpec("amazon_polarity", validation_name="test"),
    # Topic classification
    "ag_news": _DatasetSpec("ag_news", validation_name="test"),
    "dbpedia_14": _DatasetSpec("dbpedia_14", validation_name="test"),
    # NLI
    "rte": _DatasetSpec("super_glue", "rte"),
    # N.B.: We skip QNLI because we can't find the prompt templates.
    # Story completion
    "copa": _DatasetSpec("super_glue", "copa"),
    # N.B.: We skip story_cloze because it requires filling in a form to access.
    # Question answering
    "boolq": _DatasetSpec("super_glue", "boolq"),
    # Common sense reasoning
    "piqa": _DatasetSpec("piqa"),
}


def get_dlk_dataset(dataset_id: DlkDatasetId):
    dataset_spec = _DATASET_SPECS[dataset_id]
    dataset: Any = load_dataset(dataset_spec.name, dataset_spec.subset)
    return {
        **_get_dlk_dataset(dataset_id, dataset, split="train", limit=600),
        **_get_dlk_dataset(dataset_id, dataset, split="validation", limit=400),
    }


def _get_dlk_dataset(
    dataset_id: DlkDatasetId,
    dataset: Any,
    split: Split,
    limit: int,
) -> dict[str, BinaryRow]:
    dataset_spec = _DATASET_SPECS[dataset_id]
    if split == "train":
        hf_split = "train"
    elif split == "validation":
        hf_split = dataset_spec.validation_name
    else:
        raise ValueError(split)

    results = {}
    for row_idx, row in deterministic_shuffle(
        enumerate(dataset[hf_split]), lambda row: str(row[0])
    )[:limit]:
        assert "label" in row and type(row["label"]) == int, row
        template = _TEMPLATES[dataset_id]
        true_label: int = row["label"]
        false_label_options = list(range(len(template.labels)))
        false_label_options.remove(true_label)
        false_label = false_label_options[row_idx % len(false_label_options)]
        labels = sorted([true_label, false_label])

        for label_idx, label in enumerate(labels):
            format_args: dict[str, str] = dict(label=template.labels[label])
            if template.insert_label_options:
                format_args["label1"] = template.labels[labels[0]]
                format_args["label2"] = template.labels[labels[1]]
            for arg in template.args:
                format_args[arg] = row[arg]
            prompt = template.template.format(**format_args)
            results[f"{dataset_id}-{row_idx}-{label_idx}-{split}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split,
                group_id=str(row_idx),
                text=prompt,
                is_true=label == true_label,
                format_args=format_args,
                # As labels are sorted, true/false questions will be consistent.
                answer_type=str(label_idx),
            )
    return results


# Taken from <https://arxiv.org/pdf/2212.03827.pdf> appendix 1.
_TEMPLATES: dict[DlkDatasetId, _DlkTemplate] = {
    "ag_news": _DlkTemplate(
        'Consider the following example: "{text}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the topic of this example is {label}",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        args=["text"],
    ),
    "amazon_polarity": _DlkTemplate(
        'Consider the following example: "{content}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the sentiment of this example is {label}",
        labels=["Negative", "Positive"],
        args=["content"],
    ),
    "boolq": _DlkTemplate(
        "Passage: {passage}\n"
        "After reading this passage, I have a question: {question}? True or False?\n"
        "{label}",
        labels=["False", "True"],
        args=["passage", "question"],
        insert_label_options=False,
    ),
    "copa": _DlkTemplate(
        'Consider the following premise: "{premise}"\n'
        "Choice 1: {choice1}\n"
        "Choice 2: {choice2}\n"
        "Q: Which one is more likely to be the {question}, choice 1 or choice 2?\n"
        "{label}",
        labels=["Choice 1", "Choice 2"],
        args=["premise", "question", "choice1", "choice2"],
        insert_label_options=False,
    ),
    "dbpedia_14": _DlkTemplate(
        'Consider the following example: "{content}"\n'
        "Choice 1: {label1}\n"
        "Choice 2: {label2}\n"
        "Between choice 1 and choice 2, the topic of this example is {label}",
        labels=[
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ],
        args=["content"],
    ),
    "imdb": _DlkTemplate(
        'Consider the following example: "{text}"\n'
        "Between {label1} and {label2}, the sentiment of this example is {label}",
        labels=["Negative", "Positive"],
        args=["text"],
    ),
    "piqa": _DlkTemplate(
        "Goal: {goal}\n"
        "Which is the correct ending?\n"
        "Choice 1: {sol1}\n"
        "Choice 2: {sol2}\n"
        "{label}",
        labels=["Choice 1", "Choice 2"],
        args=["goal", "sol1", "sol2"],
        insert_label_options=False,
    ),
    "rte": _DlkTemplate(
        "{premise}\n"
        'Question: Does this imply that "{hypothesis}", yes or no?\n'
        "{label}",
        labels=["yes", "no"],
        args=["premise", "hypothesis"],
        insert_label_options=False,
    ),
}
