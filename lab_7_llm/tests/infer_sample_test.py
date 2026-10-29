"""
Checks the inference of the model
"""

# pylint: disable=duplicate-code, import-error
from pathlib import Path

import pytest
from torch.utils.data.dataset import Dataset

from admin_utils.constants import GLOBAL_INFERENCE_BATCH_SIZE, GLOBAL_MAX_LENGTH
from admin_utils.references.reference_scores import (
    ReferenceAnalysisScores,
    ReferenceAnalysisScoresType,
)
from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline


def run_model_inference_check(lab_path: Path, pipeline_class: type[AbstractLLMPipeline]) -> bool:
    """
    Evaluate metrics from a lab.

    Args:
        lab_path (Path): path to lab
        pipeline_class (type[AbstractLLMPipeline]): pipeline class

    Returns:
        bool: True denotes successful execution.
    """

    settings = LabSettings(lab_path / "settings.json")
    device = "cpu"

    fake_dataset = Dataset()
    pipeline = pipeline_class(
        settings.parameters.model,
        fake_dataset,
        GLOBAL_MAX_LENGTH,
        GLOBAL_INFERENCE_BATCH_SIZE,
        device,
    )
    references = ReferenceAnalysisScores(scores_type=ReferenceAnalysisScoresType.INFERENCE)
    model_name = settings.parameters.model
    for query, prediction in references.get(model_name).items():
        if "[TEST SEP]" in query:
            first_value, second_value = query.split("[TEST SEP]")
            res = pipeline.infer_sample((first_value, second_value))
        else:
            res = pipeline.infer_sample((query,))
        if res != prediction:
            assert False, (
                f"Inference of {settings.parameters.model} model is incorrect.\n"
                f"Expected: {prediction}\n"
                f"Actual: {res}"
            )
    return True


@pytest.mark.lab_7_llm
@pytest.mark.mark8
@pytest.mark.mark10
def test_analyze_ideal() -> None:
    """
    Ideal inference scenario
    """
    assert run_model_inference_check(Path(__file__).parent.parent, LLMPipeline) is True
