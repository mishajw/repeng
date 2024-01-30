from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

from repeng.datasets.elk.types import BinaryRow, DlkDatasetId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle

_LIMIT = 3000


@dataclass
class _DatasetSpec:
    name: str
    subset: str | None = None
    validation_name: str = "validation"
    use_new_line: bool = False


_DATASET_SPECS: dict[DlkDatasetId, _DatasetSpec] = {
    # Sentiment classification
    "imdb": _DatasetSpec("imdb", validation_name="test", use_new_line=True),
    "amazon_polarity": _DatasetSpec("amazon_polarity", validation_name="test"),
    # Topic classification
    "ag_news": _DatasetSpec("ag_news", validation_name="test"),
    "dbpedia_14": _DatasetSpec("dbpedia_14", validation_name="test"),
    # NLI
    "rte": _DatasetSpec("super_glue", "rte"),
    "qnli": _DatasetSpec("glue", "qnli"),
    # Story completion
    "copa": _DatasetSpec("super_glue", "copa", use_new_line=True),
    # N.B.: We skip story_cloze because it requires filling in a form to access.
    # Question answering
    "boolq": _DatasetSpec("super_glue", "boolq"),
    # Common sense reasoning
    "piqa": _DatasetSpec("piqa"),
}


def get_dlk_dataset(dataset_id: DlkDatasetId):
    dataset_spec = _DATASET_SPECS[dataset_id]
    dataset: Any = load_dataset(dataset_spec.name, dataset_spec.subset)
    templates = DatasetTemplates(
        dataset_spec.name
        if dataset_spec.subset is None
        else f"{dataset_spec.name}/{dataset_spec.subset}"
    )
    return {
        **_get_dlk_dataset(dataset_id, dataset, templates, split="train"),
        **_get_dlk_dataset(dataset_id, dataset, templates, split="validation"),
    }


def _get_dlk_dataset(
    dataset_id: DlkDatasetId,
    dataset: Any,
    templates: DatasetTemplates,
    split: Split,
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
    )[:_LIMIT]:
        assert "label" in row and type(row["label"]) == int, row
        template_names = _DATASET_TEMPLATE_NAMES[dataset_id]
        for template_name in template_names:
            template = templates.templates[templates.name_to_id_mapping[template_name]]
            answer_choice_strs = template.get_answer_choices_list(row)
            answer_choice_ints = list(range(len(answer_choice_strs)))
            for answer_choice_int in answer_choice_ints:
                prompt = _get_prompt(
                    template,
                    {**row, "label": answer_choice_int},
                )
                results[
                    f"{dataset_id}-{template_name}-{row_idx}-{answer_choice_int}"
                ] = BinaryRow(
                    dataset_id=dataset_id,
                    split="train",
                    pair_id=str(row_idx),
                    text=prompt,
                    is_true=row["label"] == answer_choice_int,
                    format_args={},
                    format_style="lat",
                    template_name=template_name,
                )
    return results


def _get_prompt(template: Template, hf_example: dict) -> str:
    prompt_pieces = template.apply(hf_example)
    assert len(prompt_pieces) == 2, (prompt_pieces, hf_example)
    question, answer = template.apply(hf_example)
    # We always separate by newline, despite the original DLK paper using spaces:
    # https://github.com/collin-burns/discovering_latent_knowledge/blob/a03dc011d0d03267c90a670cde3162f676a59fbf/utils.py#L212-L220
    # This is because a lot of the prompts we use don't make sense unless there's a
    # separator, e.g. some amazon templates put reviews and answers right next to each
    # other.
    return f"{question}\n{answer}"


# We use all the templates in promptsource, except some which fail. For reproducibility,
# we hardcode all of the template names here.
_DATASET_TEMPLATE_NAMES: dict[DlkDatasetId, list[str]] = {
    "copa": [
        "C1 or C2? premise, so/because\u2026",
        "best_option",
        "cause_effect",
        "choose",
        "exercise",
        "i_am_hesitating",
        "more likely",
    ],
    "piqa": [
        "Does this solution make sense? sol1",
        "Does this solution make sense? sol2",
        "choose the most appropriate solution",
        "finish_sentence_with_correct_choice",
        "pick_correct_choice_index",
        "pick_correct_choice_with_choice_given_before_goal",
        "what_is_the_correct_ending",
    ],
    "boolq": [
        "GPT-3 Style",
        "I wonder\u2026",
        "after_reading",
        "based on the following passage",
        "based on the previous passage",
        "could you tell me\u2026",
        "exam",
        "exercise",
        "valid_binary",
        "yes_no_question",
    ],
    "rte": [
        "GPT-3 style",
        "MNLI crowdsource",
        "based on the previous passage",
        "can we infer",
        "does it follow that",
        "does this imply",
        "guaranteed true",
        "justified in saying",
        "must be true",
        "should assume",
    ],
    "amazon_polarity": [
        "Is_this_product_review_positive",
        "Is_this_review",
        "Is_this_review_negative",
        "User_recommend_this_product",
        "convey_negative_or_positive_sentiment",
        "flattering_or_not",
        "negative_or_positive_tone",
        "user_satisfied",
        "would_you_buy",
    ],
    "imdb": [
        "Movie Expressed Sentiment",
        "Movie Expressed Sentiment 2",
        "Negation template for positive and negative",
        "Reviewer Enjoyment",
        "Reviewer Enjoyment Yes No",
        "Reviewer Expressed Sentiment",
        "Reviewer Opinion bad good choices",
        "Reviewer Sentiment Feeling",
        "Sentiment with choices ",
        "Text Expressed Sentiment",
        "Writer Expressed Sentiment",
    ],
    "qnli": [
        "based only on",
        "have all you need",
        "imply",
        "possible to answer",
        "want to know",
    ],
    "ag_news": [
        "classify",
        "classify_question_first",
        "classify_with_choices",
        "classify_with_choices_question_first",
        "recommend",
        "which_section",
        "which_section_choices",
    ],
    "dbpedia_14": [
        "given_a_choice_of_categories ",
        "given_a_list_of_category_what_does_the_title_belong_to",
        "given_list_what_category_does_the_paragraph_belong_to",
        "pick_one_category_for_the_following_text",
    ],
}
