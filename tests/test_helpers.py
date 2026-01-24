# tests for helpers
import pytest

from fairhire.helpers import colored, getenv


def test_getenv_default_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAIRHIRE_TEST_STR", raising=False)
    assert getenv("FAIRHIRE_TEST_STR", "default") == "default"


def test_getenv_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAIRHIRE_TEST_INT", "5")
    assert getenv("FAIRHIRE_TEST_INT", 1) == 5


def test_getenv_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAIRHIRE_TEST_NONE", raising=False)
    assert getenv("FAIRHIRE_TEST_NONE") is None


def test_colored_wraps() -> None:
    text = colored("hi", "red")
    assert text.startswith("\033[91m")
    assert text.endswith("\033[0m")
