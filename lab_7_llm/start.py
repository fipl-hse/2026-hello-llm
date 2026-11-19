"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import

import json
from admin_utils.constants import (
    GLOBAL_BATCH_SIZE_TEST,
    GLOBAL_MAX_LENGTH,
    DEVICE,
)
from pathlib import Path
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset
from transformers import AutoConfig

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    settings_path = Path(__file__).parent / "settings.json"

    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)

    model_name = settings["parameters"]["model"]
    dataset_name = settings["parameters"]["dataset"]

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.analyze()
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.iloc[: 100])

    pipeline = LLMPipeline(
        model_name=model_name,
        dataset=dataset,
        max_length=GLOBAL_MAX_LENGTH,
        batch_size=GLOBAL_BATCH_SIZE_TEST,
        device=DEVICE,
    )

    print(model_name)
    print(dataset_name)

    result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
