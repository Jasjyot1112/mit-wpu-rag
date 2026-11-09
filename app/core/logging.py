"""Logging configuration for the MIT-WPU RAG Engine.

Provides:
- ``setup_logging`` — configure the root / application logger.
- ``get_logger``    — retrieve a named child logger under ``rag_engine.*``.
- ``SensitiveDataFilter`` — a ``logging.Filter`` that redacts secrets from
  log records before they are emitted.
"""

from __future__ import annotations

import logging
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_LOGGER_NAME = "rag_engine"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Pattern: matches key=value or key: value sequences where the key contains a
# sensitive word.  The value portion (up to the next whitespace, comma, or
# end-of-string) is replaced with ***REDACTED***.
_SENSITIVE_PATTERN = re.compile(
    r"""
    (?P<key>                        # named group: the key / label
        (?:api[_\-]?key             # api_key / apikey / api-key
        |password
        |secret
        |token)
    )
    (?P<sep>\s*[:=]\s*)             # separator:  = or :
    (?P<value>[^\s,;'"]+)           # value: non-whitespace chars
    """,
    re.IGNORECASE | re.VERBOSE,
)
_REDACTED = r"\g<key>\g<sep>***REDACTED***"


# ---------------------------------------------------------------------------
# Sensitive data filter
# ---------------------------------------------------------------------------

class SensitiveDataFilter(logging.Filter):
    """Strip credential values from log messages before emission.

    Any log record whose formatted message contains substrings such as
    ``api_key``, ``password``, ``secret``, or ``token`` (case-insensitive)
    will have the value portion replaced with ``***REDACTED***``.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Mutate *record.msg* in-place if it contains sensitive data.

        Always returns ``True`` so the record is never silently dropped.
        """
        # Format the message first so we can inspect the final string.
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True

        redacted = _SENSITIVE_PATTERN.sub(_REDACTED, msg)
        if redacted != msg:
            # Replace the message so the handler emits the redacted version.
            record.msg = redacted
            record.args = ()  # args already consumed above

        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure the application-wide logger and return the root app logger.

    This function is idempotent — calling it multiple times is safe (handlers
    are only added once).

    Args:
        debug: When ``True`` the log level is set to ``DEBUG``; otherwise
               ``INFO``.

    Returns:
        The ``logging.Logger`` instance named ``"rag_engine"``.
    """
    level = logging.DEBUG if debug else logging.INFO

    # --- root Python logger ---------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # --- application logger ---------------------------------------------------
    app_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False  # don't double-emit to root

    # Guard: only add handler if none exist yet (idempotency).
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        app_logger.addHandler(handler)

    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped under ``rag_engine.<name>``.

    The returned logger inherits the handlers and level configured by
    ``setup_logging``.  You do **not** need to call ``setup_logging`` first —
    if the parent logger has not been configured yet the child will simply use
    Python's default WARNING level until ``setup_logging`` is called.

    Args:
        name: Logical component name, e.g. ``"retrieval"`` or ``"ingestion"``.

    Returns:
        A ``logging.Logger`` instance named ``"rag_engine.<name>"``.
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
