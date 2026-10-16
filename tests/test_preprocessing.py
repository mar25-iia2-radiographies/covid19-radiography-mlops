import cv2
import numpy as np
import pytest
from app.preprocessing import preprocess_image, make_batch


def test_preprocessing_dimensions_dtype_and_mask(tmp_path):
    image = np.full((60, 80), 200, dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[:, :40] = 255
    image_path = tmp_path / 'radiographie.png'
    mask_path = tmp_path / 'masque.png'
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    result = preprocess_image(image_path, mask_path)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255
    assert result[:, 170:, :].max() == 0
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert make_batch(result).shape == (1, 224, 224, 3)


def test_preprocessing_deterministic(tmp_path):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (60, 80), dtype=np.uint8)
    mask = np.full((60, 80), 255, dtype=np.uint8)
    a, b = tmp_path / 'image.png', tmp_path / 'mask.png'
    cv2.imwrite(str(a), image)
    cv2.imwrite(str(b), mask)
    assert np.array_equal(preprocess_image(a, b), preprocess_image(a, b))


def test_missing_image_or_mask(tmp_path):
    valid = tmp_path / 'valid.png'
    cv2.imwrite(str(valid), np.ones((8, 8), np.uint8))
    with pytest.raises(ValueError, match='radiographie'):
        preprocess_image(tmp_path / 'missing.png', valid)
    with pytest.raises(ValueError, match='masque'):
        preprocess_image(valid, tmp_path / 'missing.png')
