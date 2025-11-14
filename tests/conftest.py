"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
import shutil
import uuid
from pathlib import Path

# Add src to Python path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Test temporary directory base path (within project)
_TEST_TMP_BASE = Path(__file__).parent / ".test_tmp"


@pytest.fixture
def temp_dir(request):
    """
    Создаёт временную директорию для теста в пределах проекта.
    Директория автоматически удаляется после завершения теста.

    Все временные файлы создаются в tests/.test_tmp/ внутри проекта,
    а не в системной временной директории.
    """
    # Создаём базовую директорию, если не существует
    _TEST_TMP_BASE.mkdir(exist_ok=True)

    # Создаём уникальную поддиректорию для этого теста
    test_name = request.node.name
    # Очищаем имя от небезопасных символов для имени файла
    safe_test_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in test_name)
    unique_id = str(uuid.uuid4())[:8]
    temp_dir_path = _TEST_TMP_BASE / f"{safe_test_name}_{unique_id}"
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # Возвращаем путь
    yield temp_dir_path

    # Очистка после теста
    try:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
    except Exception as e:
        # Логируем ошибку, но не падаем
        print(f"Warning: Failed to cleanup temp directory {temp_dir_path}: {e}")


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session") 
def dataset_available():
    """Check if Common Voice 22.0 dataset is available"""
    from src.config import load_config
    from src.data import DataManager
    
    config = load_config("configs/default.yaml")
    data_manager = DataManager(config)
    return data_manager.is_dataset_available()


@pytest.fixture(scope="session")
def sample_audio_data():
    """Generate sample audio data for testing"""
    import numpy as np
    # 1 секунда аудио 16kHz
    sample_rate = 16000
    duration = 1.0
    audio_array = np.random.randn(int(sample_rate * duration))
    return {
        "array": audio_array,
        "sample_rate": sample_rate,
        "duration": duration
    }


def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line(
        "markers", "dataset_required: mark test as requiring Common Voice dataset"
    )
