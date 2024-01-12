from repeng.datasets.arc import get_arc
from repeng.datasets.common_sense_qa import get_common_sense_qa
from repeng.datasets.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.open_book_qa import get_open_book_qa
from repeng.datasets.race import get_race
from repeng.datasets.true_false import get_true_false_dataset
from repeng.datasets.truthful_qa import get_truthful_qa
from repeng.datasets.types import BinaryRow


def get_all_datasets() -> dict[str, BinaryRow | BinaryRow]:
    binary_datasets: dict[str, BinaryRow] = {
        **get_true_false_dataset(),
        **get_geometry_of_truth("cities"),
        **get_geometry_of_truth("neg_cities"),
        **get_geometry_of_truth("sp_en_trans"),
        **get_geometry_of_truth("neg_sp_en_trans"),
        **get_geometry_of_truth("larger_than"),
        **get_geometry_of_truth("smaller_than"),
        **get_geometry_of_truth("cities_cities_conj"),
        **get_geometry_of_truth("cities_cities_disj"),
    }
    paired_binary_datasets: dict[str, BinaryRow] = {
        **get_arc("challenge"),
        **get_arc("easy"),
        **get_common_sense_qa(),
        **get_open_book_qa(),
        **get_race(),
        **get_truthful_qa(),
    }
    return {**binary_datasets, **paired_binary_datasets}
