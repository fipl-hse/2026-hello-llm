"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import

import json
from pathlib import Path

from transformers import AutoConfig

from admin_utils.constants import DEVICE, GLOBAL_BATCH_SIZE_TEST, GLOBAL_MAX_LENGTH
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings_path = Path(__file__).parent / "settings.json"

    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)

    print("Mark 4: Loading of the chosen model and the chosen dataset")

    model_name = settings["parameters"]["model"]
    dataset_name = settings["parameters"]["dataset"]

    importer = RawDataImporter(dataset_name)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    if preprocessor.data is not None:
        dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        model_name=model_name,
        dataset=dataset,
        max_length=GLOBAL_MAX_LENGTH,
        batch_size=GLOBAL_BATCH_SIZE_TEST,
        device=DEVICE,
    )

    print(f"Loaded model: {model_name}")
    print(f"Loaded dataset: {dataset_name}")

    print("Mark 6: Dataset properties analysis")

    print(preprocessor.analyze())

    # reference
    # "HuggingFaceH4/no_robots": {
    #     "dataset_columns": 4,
    #     "dataset_duplicates": 0,
    #     "dataset_empty_rows": 1,
    #     "dataset_number_of_samples": 9500,
    #     "dataset_sample_max_len": 14005,
    #     "dataset_sample_min_len": 13
    # },

    result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
