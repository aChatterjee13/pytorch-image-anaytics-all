"""Generic registry pattern.

Models, backbones, datasets, and losses register via decorator so new
architectures plug in without modifying existing code:

    from image_analytics.core.registry import BACKBONES

    @BACKBONES.register("resnet50")
    class ResNet50Backbone(nn.Module): ...

    backbone = BACKBONES.build("resnet50", pretrained=True)
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, KeysView


class Registry:
    """A name -> callable mapping with decorator-based registration.

    Entries may be classes or factory functions; ``build`` simply calls the
    registered object with the provided keyword arguments.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(
        self, name: str | None = None, *, override: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a class or factory under ``name``.

        Falls back to the object's ``__name__`` when ``name`` is omitted.
        Re-registering an existing key raises unless ``override=True``.
        """

        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            key = name or obj.__name__
            if not override and key in self._entries:
                raise KeyError(
                    f"{key!r} is already registered in registry {self._name!r}; "
                    f"pass override=True to replace it"
                )
            self._entries[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<empty>"
            raise KeyError(
                f"{name!r} is not registered in registry {self._name!r}. "
                f"Available: {available}"
            ) from None

    def build(self, name: str, /, **kwargs: Any) -> Any:
        """Instantiate the entry registered under ``name`` with ``kwargs``."""
        return self.get(name)(**kwargs)

    def keys(self) -> KeysView[str]:
        return self._entries.keys()

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={sorted(self._entries)})"


# Central registries shared across the platform. Task packages re-export the
# ones they own (e.g. backbones/registry.py re-exports BACKBONES).
BACKBONES = Registry("backbones")
MODELS = Registry("models")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
NECKS = Registry("necks")
HEADS = Registry("heads")
