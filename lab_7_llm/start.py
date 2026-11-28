"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import

import json
from pathlib import Path

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

    if dataset_name is None:
        raise ValueError("Dataset name hasn't been specified")
    importer = RawDataImporter(dataset_name)
    importer.obtain()
    if importer.raw_data is None:
        raise ValueError("Raw data is None")
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    if preprocessor.data is None:
        raise ValueError("Data is None after transform")
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

    dataset_properties = preprocessor.analyze()
    print(dataset_properties)

    print(
        "Mark 8: Model inference on one sample from the dataset and on the whole dataset"
    )

    sample = dataset[0]
    print(pipeline.infer_sample(sample))
    result = pipeline.infer_dataset()

    output_path = Path(__file__).parent / "dist" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions have been saved: {output_path}")
    print(result.head())

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
