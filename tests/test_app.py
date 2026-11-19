import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from credit_scoring import MODEL_PATH  # noqa: E402


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run credit_scoring.py first")
def test_streamlit_app_predicts_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 2  # bad / good credit probabilities
