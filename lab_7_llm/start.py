"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from pathlib import Path

from admin_utils.constants import (
    DEVICE,
    GLOBAL_BATCH_SIZE_TEST,
    GLOBAL_MAX_LENGTH,
)
from core_utils.project.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time

from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / "settings.json"
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    dataset_analysis = preprocessor.analyze()
    print("Dataset analysis:")
    for property_name, value in dataset_analysis.items():
        print(f"{property_name}: {value}")
        
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=GLOBAL_MAX_LENGTH,
        batch_size=GLOBAL_BATCH_SIZE_TEST,
        device=DEVICE,
    )

    # result = None
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
