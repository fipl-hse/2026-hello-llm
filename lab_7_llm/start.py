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
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError('"importer.raw_data" is None')

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    print(preprocessor.data)
    print('----------')

    analysis = preprocessor.analyze()
    for k, v in analysis.items():
        print(f'{k}: {v}')
    print('----------')

    dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,
                           GLOBAL_MAX_LENGTH, GLOBAL_BATCH_SIZE_TEST, DEVICE)

    sample = dataset[0]
    sample_inference = pipeline.infer_sample(sample)
    print(f'{sample}: {sample_inference}')
    print('----------')

    dataset_inference = pipeline.infer_dataset()
    print(dataset_inference)
    print('----------')

    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_inference.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings['parameters']['metrics'])
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
