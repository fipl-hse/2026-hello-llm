"""
HuggingFace datasets listing.
"""

from pathlib import Path

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:
    print('Library "datasets" not installed. Failed to import.')
    load_dataset = None  # type: ignore


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Obtain dataset from HuggingFace
    dataset = load_dataset("nyu-mll/glue", name="mnli")

    # 2. Check dataset's subset
    print(dataset.keys())

    # 3. Get needed subset
    subset = dataset.get("validation_matched")
    if subset is None:
        raise ValueError("Split 'validation_matched' not found.")

    # 4. Get number of samples
    print(f"Obtained dataset step-by-step: # of samples is {len(subset)}")

    # 5. Get dataset with particular subset at once
    dataset = load_dataset("nyu-mll/glue", name="mnli", split="validation_matched")
    print(f"Obtained dataset with one call: # of samples is {len(dataset)}")

    # 6. Dataset without a name
    dataset = load_dataset("kuznetsoffandrey/sberquad", split="validation")
    print(
        f"Obtained kuznetsoffandrey/sberquad dataset with one call: # of samples is {len(dataset)}"
    )

    # 7. Cast dataset to pandas
    dataset_df: DataFrame = subset.to_pandas()

    # 8. Optionally save head of dataframe
    (
        dataset_df.head(100).to_csv(
            Path(__file__).parent / "assets" / "glue_mnli_example.csv",
            index=False,
            encoding="utf-8",
        )
    )


if __name__ == "__main__":
    main()
