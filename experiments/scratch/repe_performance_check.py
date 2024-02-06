# %%
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from jaxtyping import Float, Int64
from mppr import MContext

from repeng.activations.probe_preparations import (
    Activation,
    LabeledGroupedActivationArray,
    prepare_activations_for_probes,
)
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import DatasetId, Split
from repeng.evals.probes import eval_probe_by_question
from repeng.hooks.grab import grab
from repeng.models.llms import pythia
from repeng.probes.base import BaseGroupedProbe, BaseProbe, PredictResult
from repeng.probes.collections import ProbeMethod, train_probe

# %%
# # so we don't have to re-load massive file
# mcontext = MContext(Path("../output/comparison"))
# activations_dataset: list[ActivationResultRow] = (
#     mcontext.download_cached(
#         "activations_dataset",
#         path="s3://repeng/datasets/activations/pythia_2024-01-26_v1.pickle",
#         to="pickle",
#     )
#     .filter(lambda _, row: row.llm_id == "pythia-6.9b")
#     .get()
# )

# # %%
# mcontext = MContext(Path("../output/comparison"))
# dataset = mcontext.create_cached(
#     "dataset-v3",
#     lambda: get_datasets(["arc_easy", "common_sense_qa", "race"]),
#     to=BinaryRow,
# ).filter(
#     limit_dataset_and_split_fn(train_limit=1000, validation_limit=200),
# )

# # %%
# llm = get_llm(
#     "Llama-2-7b-chat-hf",
#     device=torch.device("cuda"),
#     dtype=torch.bfloat16,
# )

# # %%
# activations = dataset.map_cached(
#     "activations-chat",
#     lambda _, row: get_model_activations(
#         llm,
#         text=row.text,
#         last_n_tokens=3,
#     ),
#     to="pickle",
# )


# %%
mcontext = MContext(Path("../output/comparison"))
activations_dataset: list[ActivationResultRow] = mcontext.download_cached(
    "activations_dataset_llama2",
    path="s3://repeng/datasets/activations/llama-2-7b_2024-01-29_v1.pickle",
    to="pickle",
).get()


# %%
@dataclass
class ActivationSplit(Activation):
    split: Split


@dataclass
class MultipleChoiceProbe(BaseGroupedProbe):
    underlying: BaseProbe
    flip: bool

    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> "PredictResult":
        results = self.underlying.predict(activations)
        if self.flip:
            results.labels = ~results.labels
            results.logits = -results.logits
        return results

    def predict_grouped(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
        pairs: Int64[np.ndarray, "n"],  # noqa: F821
    ) -> "PredictResult":
        predictions = self.predict(activations)
        for pair in np.unique(pairs):
            logits = predictions.logits[pairs == pair]
            mean_logit = logits.mean()
            max_logit = logits.max()
            predictions.logits[pairs == pair] -= mean_logit
            predictions.labels[pairs == pair] = logits == max_logit
        return predictions


def get_group_accuracy(
    probe: BaseGroupedProbe,
    arrays: LabeledGroupedActivationArray,
) -> float:
    predict_result = probe.predict_grouped(
        arrays.activations,
        arrays.groups,
    )
    accuracy = []
    for group in np.unique(arrays.groups):
        group_labels = predict_result.labels[arrays.groups == group]
        if group_labels.sum() > 1:
            group_labels_one = np.zeros_like(group_labels)
            group_labels_one[group_labels.tolist().index(True)] = True
            group_labels = group_labels_one
        assert arrays.labels[arrays.groups == group].sum() == 1, arrays.labels[
            arrays.groups == group
        ]
        accuracy.append(
            np.all(
                predict_result.labels[arrays.groups == group]
                == arrays.labels[arrays.groups == group]
            )
        )
    return sum(accuracy) / len(accuracy)


@dataclass
class SweepSpec:
    layer: str
    token: int
    probe_method: ProbeMethod
    dataset: DatasetId
    num_samples: int


def train_and_eval_probe(spec: SweepSpec) -> dict:
    # acts = (
    #     activations_dataset.map_cached(
    #         dataset,
    #         lambda _, activations, binary_row: ActivationSplit(
    #             dataset_id=binary_row.dataset_id,
    #             pair_id=binary_row.pair_id,
    #             label=binary_row.is_true,
    #             activations=activations.activations[spec.layer][spec.token],
    #             split=binary_row.split,
    #         ),
    #     )
    #     .filter(lambda _, row: row.dataset_id == spec.dataset)
    #     .get()
    # )
    acts = [
        ActivationSplit(
            dataset_id=row.dataset_id,
            group_id=row.group_id,
            label=row.label,
            activations=row.activations[spec.layer][spec.token],
            split=row.split,
        )
        for row in activations_dataset
        if row.dataset_id == spec.dataset
    ]

    arrays = prepare_activations_for_probes(
        [row for row in acts if row.split == "train"][: spec.num_samples]
    )
    arrays_val = prepare_activations_for_probes(
        [row for row in acts if row.split == "validation"]
    )
    assert arrays_val.grouped is not None
    assert arrays_val.labeled_grouped is not None
    probe = train_probe(spec.probe_method, arrays)
    assert probe is not None
    accuracy = get_group_accuracy(
        MultipleChoiceProbe(probe, flip=False), arrays_val.labeled_grouped
    )
    accuracy_flipped = get_group_accuracy(
        MultipleChoiceProbe(probe, flip=True), arrays_val.labeled_grouped
    )
    question_eval_results = eval_probe_by_question(
        probe,
        arrays_val.labeled_grouped,
    )
    question_eval_results_mc = eval_probe_by_question(
        MultipleChoiceProbe(probe, flip=False),
        arrays_val.labeled_grouped,
    )
    return dict(
        **asdict(spec),
        accuracy=accuracy,
        accuracy_flipped=accuracy_flipped,
        question_eval_results=question_eval_results.accuracy,
        question_is_flipped=question_eval_results.is_flipped,
        question_mc_eval_results=question_eval_results_mc.accuracy,
        question_mc_is_flipped=question_eval_results_mc.is_flipped,
    )


probe_methods: list[ProbeMethod] = ["lr", "lr-grouped", "lat"]
specs = mcontext.create(
    {
        f"{layer}-{token}-{probe_method}-{dataset_id}-{num_samples}": SweepSpec(
            layer, token, probe_method, dataset_id, num_samples
        )
        for layer in [f"h{i}" for i in range(28, 32)]
        for token in range(3)
        for probe_method in probe_methods
        for dataset_id, num_samples in [
            ("arc_easy", 25 * 5),
            # ("arc_easy", 50 * 5),
            # ("arc_easy", 100 * 5),
            # ("common_sense_qa", 7 * 4),
            # ("common_sense_qa", 14 * 4),
            # ("common_sense_qa", 100 * 4),
            # ("race", 3 * 4),
            # ("race", 6 * 4),
            # ("race", 100 * 4),
        ]
    }
)

probe_results = specs.map_cached(
    "probe-results-v4",
    lambda _, spec: train_and_eval_probe(spec),
    to="pickle",
)

# for layer in ["h17", "h18", "h19", "h20"]:
#     pprint(
#         train_and_eval_probe(
#             SweepSpec(
#                 layer=layer,
#                 token=2,
#                 probe_method="lat",
#                 dataset="arc_easy",
#                 num_samples=101 * 5,
#             )
#         )
#     )

# %%
df = probe_results.to_dataframe(lambda d: d)
df["accuracy"] = df.apply(
    lambda row: max(row["accuracy"], row["accuracy_flipped"]), axis=1
)
df = df[df["dataset"] == "arc_easy"]
g = sns.FacetGrid(df, col="probe_method", row="token")
g = g.map(sns.lineplot, "layer", "question_eval_results", "num_samples", marker="o")
g.add_legend()
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# %%
df.groupby(["dataset", "probe_method"])["accuracy"].max()

# %%
llm = pythia("pythia-70m", device=torch.device("cpu"), dtype=torch.float32)

# %%
toks = llm.tokenizer.encode("hello world", return_tensors="pt")
with torch.no_grad():
    point_idx = -6
    with grab(llm.model, point=llm.points[point_idx]) as grab_fn:
        output = llm.model(toks, output_hidden_states=True)
        print(output.hidden_states[point_idx].flatten()[:5].numpy())
        print(grab_fn().flatten()[:5].numpy())

# %%
hs = output.hidden_states
print(len(hs))
print([h.shape for h in hs])
