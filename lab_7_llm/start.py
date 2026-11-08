"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from pathlib import Path

from admin_utils.constants import DEVICE, GLOBAL_BATCH_SIZE_TEST, GLOBAL_MAX_LENGTH
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        settings.parameters.model,
        dataset,
        GLOBAL_MAX_LENGTH,
        GLOBAL_BATCH_SIZE_TEST,
        DEVICE,
    )

    result = pipeline
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()