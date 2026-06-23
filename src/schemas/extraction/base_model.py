"""
Base extraction model with auto-injected field validators.

Architecture:
    ProfileValidatorMixin  (__init_subclass__ hook)
        │
        ├── BaseExtractionModel (BaseModel + Mixin)
        │       ├── DividendExtraction     ← ValidationProfile = DividendValidationProfile
        │       └── MandaExtraction        ← ValidationProfile = MandaValidationProfile
        │
        └── BaseDoc subclasses (e.g. DividendExtractionResult)
                └── also uses the same ValidationProfile via Mixin
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


def _make_allowed_validator(field_name: str, allowed_attr: str) -> classmethod:
    """
    Factory: creates a @field_validator that checks allowed values at runtime.

    The validator looks up ``cls.ValidationProfile.<allowed_attr>`` **at call
    time**, which naturally supports Profile inheritance — a subclass Profile
    that overrides the set will be picked up without any extra wiring.

    Each factory call creates an independent closure scope, preventing the
    classic Python closure-capture trap.
    """

    @field_validator(field_name)
    @classmethod
    def validator(cls, v: Any) -> Any:
        if v is None:
            return v
        profile = getattr(cls, "ValidationProfile", None)
        if profile is None:
            return v
        allowed = getattr(profile, allowed_attr, None)
        if allowed is None:
            return v
        if v not in allowed:
            raise ValueError(f"'{field_name}' = {v!r} not in allowed set: {allowed}")
        return v

    # Unique name prevents Pydantic from deduplicating across subclasses
    validator.__name__ = f"_validate_{field_name}"
    return validator


def _inject_date_validator(cls: type, field_name: str) -> None:
    """
    Inject a ``YYYY-MM-DD`` format validator for one ``*_date`` field.

    Called automatically by ``ProfileValidatorMixin.__init_subclass__``
    for every field whose name ends in ``_date``.
    """

    @field_validator(field_name)
    @classmethod
    def _date_validator(cls, v: Any) -> Any:
        if v is None:
            return v
        import re

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Date field '{field_name}' must be YYYY-MM-DD, got '{v}'")
        return v

    _date_validator.__name__ = f"_validate_{field_name}"
    setattr(cls, _date_validator.__name__, _date_validator)


class ProfileValidatorMixin:
    """
    Mixin that auto-injects allowed-value validators from ValidationProfile.

    Works with any Pydantic model (BaseModel or BaseModel subclasses).
    The subclass sets ``ValidationProfile`` and validators are injected
    via ``__init_subclass__`` — no manual ``@field_validator`` needed.

    Example::

        class MyModel(BaseModel, ProfileValidatorMixin):
            ValidationProfile = MyProfile
            currency: Optional[str] = Field(None)
    """

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # ── 1) Auto-inject ALLOWED_* validators from ValidationProfile ──
        profile = getattr(cls, "ValidationProfile", None)
        if profile is not None:
            _field_map = {
                "ALLOWED_CURRENCIES": "currency",
                "ALLOWED_DIVIDEND_TYPES": "dividend_type",
                "ALLOWED_FREQUENCIES": "frequency",
                "ALLOWED_PAYMENT_METHODS": "payment_method",
            }
            for attr_name in dir(profile):
                if not attr_name.startswith("ALLOWED_"):
                    continue
                pydantic_field = _field_map.get(attr_name)
                if pydantic_field is None:
                    pydantic_field = attr_name.removeprefix("ALLOWED_").lower()

                validator = _make_allowed_validator(pydantic_field, attr_name)
                setattr(cls, validator.__name__, validator)

        # ── 2) Convention: any *_date field gets YYYY-MM-DD format check ──
        for field_name in cls.__annotations__:
            if field_name.endswith("_date"):
                _inject_date_validator(cls, field_name)


class BaseExtractionModel(BaseModel, ProfileValidatorMixin):
    """
    Pydantic BaseModel that auto-injects allowed-value validators.

    Subclasses set ``ValidationProfile`` to a class that derives from
    ``BaseValidationProfile``. The validators are injected automatically
    via ``__init_subclass__`` — no manual ``@field_validator`` needed.
    """

    pass
