"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json

from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from admin_utils.constants import GLOBAL_MAX_LENGTH, GLOBAL_BATCH_SIZE_TEST, DEVICE


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    print(preprocessor.data)
    analysis = preprocessor.analyze()
    for k, v in analysis.items():
        print(f'{k}: {v}')

    dataset = TaskDataset(preprocessor.data.iloc[: 5])  # TODO: change to 100 before submission

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,
                           GLOBAL_MAX_LENGTH, GLOBAL_BATCH_SIZE_TEST, DEVICE)

    sample = dataset[0]
    sample_inference = pipeline.infer_sample(sample)
    print(f'{sample}: {sample_inference}')

    dataset_inference = pipeline.infer_dataset()
    # print(dataset_inference)

    dataset_inference_path = Path(__file__).parent.parent / 'dist' / 'predictions.csv'
    dataset_inference_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_inference.to_csv(dataset_inference_path, index=False)

    result = dataset_inference
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
