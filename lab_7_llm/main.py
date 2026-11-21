"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import json
from pathlib import Path
from typing import Iterable, Sequence

import datasets
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline, HFModelLike
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import (
    AbstractRawDataPreprocessor,
    ColumnNames,
)
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

        if isinstance(self._hf_name, str) and self._hf_name is not None:
            df = datasets.load_dataset(path=self._hf_name, split="train").to_pandas()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Downloaded dataset is not pd.DataFrame")
        self._raw_data = df


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

        raw_data = self._raw_data
        source_column = "prompt"

        key_parameters = {
            "dataset_number_of_samples": -1,
            "dataset_columns": -1,
            "dataset_duplicates": -1,
            "dataset_empty_rows": -1,
            "dataset_sample_min_len": -1,
            "dataset_sample_max_len": -1,
        }

        key_parameters["dataset_number_of_samples"] = int(len(raw_data))

        key_parameters["dataset_columns"] = int(len(raw_data.columns))

        row_signatures = raw_data.apply(
            lambda row: json.dumps(
                row.to_dict(), sort_keys=True, default=str, ensure_ascii=False
            ),
            axis=1,
        )

        key_parameters["dataset_duplicates"] = int(row_signatures.duplicated().sum())

        empty_rows = raw_data[source_column].isna() | raw_data[source_column].astype(
            str
        ).str.strip().eq("")
        dataset_empty_rows = int(empty_rows.sum())
        key_parameters["dataset_empty_rows"] = dataset_empty_rows

        clean_data = raw_data.loc[~empty_rows]

        prompt_lengths = clean_data["prompt"].astype(str).str.len()

        message_lengths = clean_data["messages"].apply(
            lambda messages: sum(
                len(str(message.get("content", "")))
                for message in messages
                if isinstance(message, dict)
            )
        )

        lengths = pd.concat([prompt_lengths, message_lengths])

        key_parameters["dataset_sample_max_len"] = (
            int(lengths.max()) if not lengths.empty else 0
        )

        key_parameters["dataset_sample_min_len"] = (
            int(lengths.min()) if not lengths.empty else 0
        )

        return key_parameters

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        df = self._raw_data.copy()

        df = df[df["category"] == "Closed QA"]

        df = df[["prompt", "messages"]].copy()

        df = df.rename(columns={"prompt": "question"})

        df[ColumnNames.CONTEXT.value] = df["messages"].apply(
            lambda messages: messages[0]["content"]
        )

        df[ColumnNames.TARGET.value] = df["messages"].apply(
            lambda messages: messages[1]["content"]
        )

        df = df.drop(columns=["messages"])

        df = df.reset_index(drop=True)

        self._data = df


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
        self._data = data.head(100)

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
        return tuple(self._data.iloc[index].values)

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
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, dtype=torch.float32)
        model.to(self._device)
        model.eval()
        self._model = model

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
        return pd.DataFrame()

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        return list("")


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
        return dict()
