import pytest

from image_analytics.core.registry import Registry


def test_register_and_build():
    reg = Registry("test")

    @reg.register("widget")
    class Widget:
        def __init__(self, size=1):
            self.size = size

    assert "widget" in reg
    assert len(reg) == 1
    built = reg.build("widget", size=5)
    assert isinstance(built, Widget)
    assert built.size == 5


def test_register_default_name():
    reg = Registry("test")

    @reg.register()
    def my_factory():
        return 42

    assert "my_factory" in reg
    assert reg.build("my_factory") == 42


def test_duplicate_registration_raises():
    reg = Registry("test")
    reg.register("dup")(lambda: 1)
    with pytest.raises(KeyError, match="already registered"):
        reg.register("dup")(lambda: 2)


def test_duplicate_with_override():
    reg = Registry("test")
    reg.register("dup")(lambda: 1)
    reg.register("dup", override=True)(lambda: 2)
    assert reg.build("dup") == 2


def test_missing_key_lists_available():
    reg = Registry("test")
    reg.register("known")(lambda: 1)
    with pytest.raises(KeyError, match="known"):
        reg.get("unknown")
