from collections.abc import Iterable
from typing import Any


class ContractValidationError(ValueError):
    """Structured failure raised when a contract cannot be trusted."""

    reason_code = "CONTRACT_VALIDATION_FAILED"

    def __init__(self, errors: Iterable[dict[str, Any]]) -> None:
        self.errors = [
            {
                "loc": list(error.get("loc", ())),
                "type": error.get("type", "value_error"),
                "msg": error.get("msg", "invalid contract value"),
            }
            for error in errors
        ]
        super().__init__(self.reason_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "errors": self.errors,
        }


class ContractParseError(ValueError):
    """Structured failure raised before schema validation can begin."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        path: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.path = path
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "path": self.path,
        }
