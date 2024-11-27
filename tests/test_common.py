import pytest

from survivalist.base import SurvivalAnalysisMixin
from survivalist.testing import all_survival_estimators


@pytest.mark.parametrize("estimator_cls", all_survival_estimators())
def test_survival_analysis_base_clas(estimator_cls):
    assert hasattr(estimator_cls, "fit")
    assert hasattr(estimator_cls, "predict")
    assert hasattr(estimator_cls, "score")
    assert issubclass(estimator_cls, SurvivalAnalysisMixin)
