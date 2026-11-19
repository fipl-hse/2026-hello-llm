"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    hf_name = 'HuggingFaceH4/no_robots'
    importer = RawDataImporter(hf_name)
    preprocessor = RawDataPreprocessor(importer.obtain())
    importer.obtain()
    print(importer.raw_data.head())
    # result = None
    print("Hello, world!")
    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
