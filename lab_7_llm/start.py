"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from main import RawDataImporter, RawDataPreprocessor, TaskDataset

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    importer = RawDataImporter(settings.parameters.dataset)
    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset = TaskDataset(preprocessor.data.iloc[: 100])

    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
