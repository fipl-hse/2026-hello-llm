"""
HuggingFace evaluate listing.
"""

# pylint: disable=duplicate-code


try:
    import torch
    from torch import argmax
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore

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

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from evaluate import load
except ImportError:
    print('Library "evaluate" not installed. Failed to import.')


class TaskDataset(Dataset):  # type: ignore
    """
    Dataset with classification data.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): original data.
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> str:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            str: The item to be received
        """
        # The "neutral" column contains the raw texts which may be toxic
        return str(self._data["neutral"].iloc[index])


def run_classification_scenario() -> None:
    """
    Run toxicity classification scenario.
    """

    # 1. Load dataset
    data = load_dataset("s-nlp/ru_paradetox_toxicity", split="train").to_pandas()
    dataset = TaskDataset(data.iloc[:100])
    references = data["toxic"].iloc[:100].astype(int).tolist()

    # 2. Get data loader with batch 4
    dataset_loader = DataLoader(dataset, batch_size=4)
    print(len(dataset_loader))

    # 3. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("khvatov/ru_toxicity_detector")
    model = AutoModelForSequenceClassification.from_pretrained("khvatov/ru_toxicity_detector")

    # 4. Inference dataset
    predictions = []
    with torch.no_grad():
        for batch_data in dataset_loader:
            ids = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
            output = model(**ids).logits
            predictions.extend(list(argmax(output, dim=1)))

    # 5. Print predictions
    print("Predictions:", predictions)
    print("References:", references)

    # 6. Load metric
    accuracy_metric = load("accuracy")
    print("Metric name:", accuracy_metric.name)

    # 7. Compute accuracy
    print(accuracy_metric.compute(references=references, predictions=predictions))


def run_generation_scenario() -> None:
    """
    Run text generation scenario.
    """

    prompts = [
        "Natural Language Processing is",
        "The capital of France is",
    ]
    # Gold reference continuations for evaluation
    gen_references = [
        ["a field of artificial intelligence."],
        ["Paris, the city of light."],
    ]

    # 8. Load model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Ensure padding token is set for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 9. Tokenize batch
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")

    # 10. Generate sequences
    with torch.no_grad():
        output_ids = model.generate(
            **tokens,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 11. Decode generated tokens to string
    prompt_len = tokens["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    gen_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Predictions:", gen_predictions)

    # 12. Evaluate with BLEU
    bleu_metric = load("bleu")
    result = bleu_metric.compute(predictions=gen_predictions, references=gen_references)
    print("BLEU Result:", result["bleu"])


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # Classification scenario
    run_classification_scenario()

    # Generation scenario
    run_generation_scenario()


if __name__ == "__main__":
    main()
