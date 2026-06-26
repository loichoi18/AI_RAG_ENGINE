from __future__ import annotations
from collections.abc import Iterator
import pytest
from configs.settings import Settings, get_settings


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: tests requiring a live Qdrant (skipped when unreachable)")


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear(); yield; get_settings.cache_clear()
