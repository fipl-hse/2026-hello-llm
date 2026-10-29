"""
Checks that the dataset is being analyzed correctly
"""

# pylint: disable=duplicate-code, assignment-from-no-return
from pathlib import Path

import pytest

from admin_utils.references.reference_scores import ReferenceAnalysisScores
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from lab_7_llm.start import main


@pytest.mark.lab_7_llm
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_analyze_ideal() -> None:
    """
    Ideal analyze scenario
    """
    main()

    settings = LabSettings(Path(__file__).parent.parent / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError('"importer.raw_data" can not be None!')
    preprocessor = RawDataPreprocessor(importer.raw_data)

    dataset_analysis = preprocessor.analyze()

    references = ReferenceAnalysisScores()

    assert references.get(settings.parameters.dataset) == dataset_analysis

    assert len(dataset_analysis) == 6
