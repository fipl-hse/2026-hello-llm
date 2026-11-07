"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from pathlib import Path

from admin_utils.constants import DEVICE, GLOBAL_BATCH_SIZE_TEST, GLOBAL_MAX_LENGTH
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)

CURRENT_DIR = Path(__file__).parent
SETTINGS_PATH = CURRENT_DIR / "settings.json"
PREDS_PATH = CURRENT_DIR / "dist" / "predictions.csv"


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(SETTINGS_PATH)
    raw_data_importer = RawDataImporter(hf_name=settings.parameters.dataset)
    raw_data_importer.obtain()

    assert (
        raw_data := raw_data_importer.raw_data
    ) is not None, "Raw data importer does not work correctly"
    raw_data_preprocessor = RawDataPreprocessor(raw_data=raw_data)
    print(raw_data_preprocessor.analyze())
    raw_data_preprocessor.transform()

    dataset = TaskDataset(raw_data_preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=GLOBAL_MAX_LENGTH,
        batch_size=GLOBAL_BATCH_SIZE_TEST,
        device=DEVICE,
    )

    result = pipeline.infer_dataset()
    PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(PREDS_PATH, index=False)

    task_evaluator = TaskEvaluator(data_path=PREDS_PATH, metrics=settings.parameters.metrics)

    print(f"Inference metrics: {task_evaluator.run()}")

    assert result is not None, "Pipeline does not work correctly"


if __name__ == "__main__":
    main()
