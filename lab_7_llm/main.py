"""
Laboratory work.

Working with Large Language Models.
"""
from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if not isinstance(raw_data := load_dataset(self._hf_name, split="test").to_pandas(), 
                          pd.DataFrame):
            raise TypeError("Downloaded dataset is not right type.")

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
        data = self._raw_data
        empty_rows = int(data.isna().all(axis=1).sum())
        non_empty = data.dropna(how="all")
        lengths = non_empty["instruction"].str.len()
        return {
            "dataset_number_of_samples": len(data),
            "dataset_columns": len(data.columns),
            "dataset_duplicates": int(data.duplicated().sum()),
            "dataset_empty_rows": empty_rows,
            "dataset_sample_min_len": int(lengths.min()),
            "dataset_sample_max_len": int(lengths.max()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (
        self._raw_data
        .drop(columns=["context", "category", "text"])
        .rename(
            columns={
                "instruction": ColumnNames.QUESTION.value,
                "response": ColumnNames.TARGET.value,
            }
        )
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
        row = self._data.iloc[index]
        question = str(row[ColumnNames.QUESTION.value])
        return (question,)

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
        self._model_name = model_name
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model.eval()
        self._model.to(self._device)

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """


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

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
