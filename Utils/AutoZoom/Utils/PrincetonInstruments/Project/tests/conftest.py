"""pytest 共享配置。"""

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 PYTHONPATH 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_backend():
    from pi_spectrometer.picam.demo import MockSpectrometerBackend
    return MockSpectrometerBackend()


@pytest.fixture
def sample_spectrum():
    import numpy as np
    x = np.linspace(466, 566, 1024)
    y = 1000.0 * np.exp(-0.5 * ((x - 516.0) / 18.0) ** 2)
    y += np.random.normal(0, 10, size=x.shape)
    return x, y
