"""
Classification (NLI) task listing.
"""

# pylint: disable=duplicate-code, too-many-locals, unused-argument

import time
from pathlib import Path
from typing import Any

try:
    from pandas import DataFrame, read_csv
except ImportError:
    print('Library "pandas" not installed. Failed to import.')

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


class NLIDataset(Dataset):
    """
    TaskDataset returning pairs of sentences for NLI.
    """

    def __init__(self, data: DataFrame) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, str]:
        row = self._data.iloc[index]
        return str(row["premise"]), str(row["hypothesis"])


def inspect_tokenizer_and_model(
    tokenizer: Any,
    model: Any,
) -> None:
    """
    Demonstrates tokenization, special tokens, unpacking and model head.
    """

    sample_text = "There's a silencing as the machine arrives here."

    # 1. Subword splitting
    tokens = tokenizer.tokenize(sample_text)
    print(f"Sample text: '{sample_text}'")
    print(f"Subwords: {tokens}")

    # 2. Token IDs and special tokens
    encoded = tokenizer(sample_text)
    print(f"Input IDs with special tokens: {encoded['input_ids']}")
    print(f"Decoded back: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")

    # 3. Unpacking
    # tokenizer(...) returns a BatchEncoding (dict-like object)
    print(f"Dictionary keys returned by tokenizer: {list(encoded.keys())}")
    print("Passing **tokens unpacks {'input_ids': ..., 'attention_mask': ...}")
    print("directly into the matching keyword arguments of model.forward()!\n")

    # 4. Model head inspection
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Classification head: {model.classifier}")
    print(f"id2label mapping: {model.config.id2label}\n")


def main() -> None:
    """
    Entrypoint for the listing.
    """

    model_name = "cointegrated/rubert-tiny-bilingual-nli"
    data_path = Path(__file__).parent / "assets" / "glue_mnli_example.csv"

    # 5. Load tokenizer and classification model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    # 6. Inspect internals
    inspect_tokenizer_and_model(tokenizer, model)

    # 7. Load data
    df = read_csv(data_path)
    dataset = NLIDataset(df)
    print(f"Loaded {len(dataset)} samples from {data_path.name}\n")

    # 8. Prepare dataloader
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Running Inference with batch_size={batch_size}")
    print(f"Total batches to process: {len(loader)}\n")

    # 9. Get predictions
    all_predictions = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (batch_premises, batch_hypotheses) in enumerate(loader):
            # Pre-processing: Dynamic padding within this specific batch
            tokens = tokenizer(
                list(batch_premises),
                list(batch_hypotheses),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Observe batch shapes
            current_batch_len = len(batch_premises)
            seq_len = tokens["input_ids"].shape[1]
            print(f"Batch {batch_idx + 1}: size={current_batch_len} | padded_length={seq_len}")

            outputs = model(**tokens)
            batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
            all_predictions.extend([model.config.id2label[p] for p in batch_preds])

    print(f"Finished in {time.time() - start_time:.3f} seconds ({len(all_predictions)} samples)\n")

    # 10. Attach predictions back to original DataFrame
    df["predictions"] = all_predictions
    print("Predictions sample (first 3 rows):")
    print(df[["premise", "hypothesis", "predictions"]].head(3))


# TASKS


# Task 1:
def extract_predictions(
    logits: torch.Tensor,
    id2label: dict[int, str],
) -> list[str]:
    """
    Extract human-readable labels from model logits using argmax.

    Args:
        logits (torch.Tensor): Model output logits of shape [batch_size, num_classes].
        id2label (dict[int, str]): Mapping from class index to label string.

    Returns:
        list[str]: Predicted class labels for each sample in the batch.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# extract_predictions(torch.tensor([[0.2, 1.8], [2.1, -0.5]]), \
# {0: "neutral", 1: "toxic"}) -> ['toxic', 'neutral']
# extract_predictions(torch.tensor([[-1.0, 0.0, 2.5]]), \
# {0: "entailment", 1: "neutral", 2: "contradiction"}) -> ['contradiction']


# Task 2:
def unzip_nli_batch(
    sample_batch: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """
    Unzip a batch of (premise, hypothesis) tuples yielded by DataLoader
    into two separate lists suitable for passing into tokenizer(premises, hypotheses).

    Args:
        sample_batch (list[tuple[str, str]]): List of tuples
        where each tuple is (premise, hypothesis).

    Returns:
        tuple[list[str], list[str]]: Tuple containing (list_of_premises, list_of_hypotheses).
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# unzip_nli_batch([("P1", "H1"), ("P2", "H2")]) -> (['P1', 'P2'], ['H1', 'H2'])
# unzip_nli_batch([("A", "B")]) -> (['A'], ['B'])


# Task 3:
def transform_ids_to_tokens(
    token_ids: list[int],
    tokenizer: AutoTokenizer,
) -> list[str]:
    """
    Convert a list of numerical token IDs back into their string token representations.

    Args:
        token_ids (list[int]): List of token IDs.
        tokenizer (AutoTokenizer): Hugging Face tokenizer instance.

    Returns:
        list[str]: List of token strings.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# transform_ids_to_tokens([101, 2054, 102], tokenizer) -> ['[CLS]', 'what', '[SEP]']


# Task 4:
def get_batch_padded_length(tokens: dict[str, torch.Tensor]) -> int:
    """
    Get the sequence length (padded length) of the current tokenized batch.

    Args:
        tokens (dict[str, torch.Tensor]): Dictionary produced by tokenizer with 'input_ids'.

    Returns:
        int: Length of the padded sequence dimension.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# get_batch_padded_length({"input_ids": torch.zeros((4, 28), dtype=torch.long)}) -> 28
# get_batch_padded_length({"input_ids": torch.zeros((1, 15), dtype=torch.long)}) -> 15


# Task 5:
def attach_predictions_to_df(
    df: DataFrame,
    predictions: list[str],
) -> DataFrame:
    """
    Attach model predictions as a new column named 'predictions' to a copy of the input DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.
        predictions (list[str]): List of predicted labels matching DataFrame length.

    Returns:
        DataFrame: New DataFrame containing the added 'predictions' column.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# attach_predictions_to_df(DataFrame({"text": ["A", "B"]}), \
# ["pos", "neg"])["predictions"].tolist() -> ['pos', 'neg']


# Task 6:
def unwrap_single_prediction(batch_predictions: list[str]) -> str:
    """
    Extract a single prediction string from a batch result.

    Args:
        batch_predictions (list[str]): List of predictions returned by batch inference.

    Returns:
        str: The first prediction in the batch.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# unwrap_single_prediction(["entailment"]) -> 'entailment'
# unwrap_single_prediction(["toxic", "neutral"]) -> 'toxic'


if __name__ == "__main__":
    main()
