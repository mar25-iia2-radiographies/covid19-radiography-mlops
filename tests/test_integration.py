"""Tests d'intégration : preprocessing réel -> modèle VGG16 -> probabilités."""
from pathlib import Path
import numpy as np
import pytest

from app.model import CLASS_NAMES, load_covid_model, predict_image
from app.preprocessing import make_batch, preprocess_image

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data" / "samples" / "covid_test.png"
MASK = ROOT / "data" / "samples" / "covid_mask.png"

# Valeurs de référence du golden test, arrondies à 8 décimales.
REFERENCE = np.array(
    [0.05521444, 0.00001736, 0.94476587, 0.00000231],
    dtype=np.float64,
)

@pytest.fixture(scope="module")
def probabilities():
    if not IMAGE.is_file() or not MASK.is_file():
        pytest.fail("Échantillons manquants dans data/samples : covid_test.png et covid_mask.png")
    image = preprocess_image(IMAGE, MASK)
    batch = make_batch(image)
    model = load_covid_model()
    return np.asarray(predict_image(model, batch), dtype=np.float64)


def test_output_contract(probabilities):
    assert probabilities.shape == (4,)
    assert len(CLASS_NAMES) == 4
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)


def test_golden_prediction(probabilities):
    # Tolérance couvrant de faibles écarts numériques entre exécutions CPU.
    np.testing.assert_allclose(probabilities, REFERENCE, atol=1e-4, rtol=1e-4)


def test_known_misclassification_is_reproducible(probabilities):
    """Ce test fige le comportement existant, pas une vérité médicale."""
    assert CLASS_NAMES[int(np.argmax(probabilities))] == "Lung_Opacity"
