"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from main import RawDataImporter, RawDataPreprocessor, TaskDataset, LLMPipeline
from admin_utils.constants import GLOBAL_BATCH_SIZE_TEST, GLOBAL_MAX_LENGTH, DEVICE
from core_utils.llm.time_decorator import report_time


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    with open('settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.iloc[: 100])

    batch_size = GLOBAL_BATCH_SIZE_TEST
    max_length = GLOBAL_MAX_LENGTH
    device = DEVICE

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, max_length, batch_size, device)

    result = (type(pipeline),
              type(dataset))
    print(dataset)
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
