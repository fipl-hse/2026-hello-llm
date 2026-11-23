"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from pathlib import Path

from admin_utils import constants
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
)


def print_data_info(data_info) -> None:
    print()
    for key, value in data_info.items():
        print(f"{key}: {value}")
    print()


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    preprocessor = RawDataPreprocessor(importer.raw_data)
    data_info = preprocessor.analyze()
    print_data_info(data_info)
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=constants.GLOBAL_MAX_LENGTH,
        batch_size=constants.GLOBAL_BATCH_SIZE_TEST,
        device=constants.DEVICE,
    )

    result = pipeline
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
