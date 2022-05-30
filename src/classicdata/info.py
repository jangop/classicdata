"""
Information about the implemented datasets.
"""
import warnings

import tabulate

from .dataset import CitationWarning, Dataset, PublicDataset


def list_datasets(table_format: str) -> str:
    """
    List all implemented public datasets in form of a table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CitationWarning)
        all_datasets = [
            implementation() for implementation in PublicDataset.__subclasses__()
        ]

    def format_dataset(dataset: Dataset, link: bool) -> dict:
        name = (
            f"[{dataset.short_name}]({dataset.source.url})"
            if link
            else dataset.short_name
        )
        entry = {
            "Dataset": name,
            "# Samples": dataset.n_samples,
            "# Features": dataset.n_features,
            "Feature Type": dataset.feature_type.name.lower(),
            "# Classes": dataset.n_classes,
            "Balance": dataset.balance,
        }
        return entry

    data = [
        format_dataset(dataset, link=table_format == "github")
        for dataset in all_datasets
    ]
    return tabulate.tabulate(
        data, headers="keys", tablefmt=table_format, floatfmt=".2f"
    )


def cli():
    """
    Command Line Interface to list datasets
    """
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser()
    parser.add_argument("format", type=str, nargs="?", default="simple")
    args = parser.parse_args()

    print(list_datasets(table_format=args.format))


if __name__ == "__main__":
    cli()
