"""
Laboratory work.

Working with Large Language Models.
"""

from ast import literal_eval
from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Any, cast, Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        if not isinstance(
            raw_data := load_dataset(self._hf_name, split="val_en").to_pandas(),
            pd.DataFrame,
        ):
            raise TypeError(f"Downloaded dataset has inappropriate type: {type(raw_data)}")

        self._raw_data = raw_data


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        data = self._raw_data.copy()
        data["tokens"] = data["tokens"].map(tuple)
        data["ner_tags"] = data["ner_tags"].map(tuple)
        lengths = data["tokens"].map(len)
        return {
            "dataset_columns": len(data.columns),
            "dataset_duplicates": int(data.duplicated().sum()),
            "dataset_empty_rows": int(data.isna().any(axis=1).sum()),
            "dataset_number_of_samples": len(data),
            "dataset_sample_max_len": int(lengths.max()),
            "dataset_sample_min_len": int(lengths.min()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = pd.DataFrame(
            {
                ColumnNames.SOURCE.value: self._raw_data["tokens"].map(" ".join),
                ColumnNames.TARGET.value: self._raw_data["ner_tags"].map(
                    lambda tags: str(tags.tolist())
                ),
            }
        )


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return (self._data.iloc[index][ColumnNames.SOURCE.value],)

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if self._model is None:
            return None

        return cast(str, self._infer_batch([sample])[0])

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        loader = DataLoader(
            self._dataset, batch_size=self._batch_size, shuffle=False, collate_fn=list
        )
        preds = []

        for batch in loader:
            preds.extend(self._infer_batch(batch))

        return DataFrame(
            {
                ColumnNames.TARGET.value: self._dataset.data[ColumnNames.TARGET.value],
                ColumnNames.PREDICTION.value: preds,
            }
        )

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            return []
        words = [sample[0].split() for sample in sample_batch]

        inputs = self._tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        results = []

        for batch_index, token_preds in enumerate(preds):
            word_ids = inputs.word_ids(batch_index=batch_index)
            word_preds = []
            previous_word_id = None

            for word_id, prediction in zip(word_ids, token_preds):
                if word_id is not None and word_id != previous_word_id:
                    word_preds.append(prediction)

                previous_word_id = word_id

            results.append(str(word_preds))

        return results


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        refs = []
        preds = []

        for reference, prediction in zip(
            data[ColumnNames.TARGET.value], data[ColumnNames.PREDICTION.value]
        ):
            refs.extend(literal_eval(reference))
            preds.extend(literal_eval(prediction))

        accuracy = load(next(iter(self._metrics)).value)

        return cast(dict[Any, Any], accuracy.compute(references=refs, predictions=preds))
