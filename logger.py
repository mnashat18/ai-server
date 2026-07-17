import logging
import os
import sys
import time


_OWNED_HANDLER_ATTR = "_ai_server_owned_handler"
_LEGACY_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class _UtcFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # pragma: no cover - exercised via tests
        dt = time.gmtime(record.created)
        if datefmt:
            return time.strftime(datefmt, dt)
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", dt)


def _validate_logger_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("logger name must be a non-empty string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("logger name must be a non-empty string")
    return normalized


def _parse_log_level() -> int:
    raw = os.getenv("LOG_LEVEL", "INFO")
    if not isinstance(raw, str):
        raise ValueError("LOG_LEVEL must be a string")
    normalized = raw.strip().upper()
    if not normalized:
        raise ValueError("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, WARN, FATAL")
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    level = level_map.get(normalized)
    if level is None:
        raise ValueError("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, WARN, FATAL")
    return level


def _is_owned_handler(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _OWNED_HANDLER_ATTR, False))


def _is_legacy_owned_handler(handler: logging.Handler) -> bool:
    formatter = getattr(handler, "formatter", None)
    return (
        isinstance(handler, logging.StreamHandler)
        and not _is_owned_handler(handler)
        and handler.level == logging.NOTSET
        and getattr(handler, "stream", None) is sys.stderr
        and not handler.filters
        and formatter is not None
        and getattr(formatter, "_fmt", None) == _LEGACY_FORMAT
    )


def _find_owned_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if _is_owned_handler(handler):
            return handler
    return None


def _adopt_legacy_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if _is_legacy_owned_handler(handler):
            setattr(handler, _OWNED_HANDLER_ATTR, True)
            return handler
    return None


def get_logger(name: str = "ai-server") -> logging.Logger:
    logger_name = _validate_logger_name(name)
    level = _parse_log_level()
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    handler = _find_owned_handler(logger)
    if handler is None:
        handler = _adopt_legacy_handler(logger)
    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _OWNED_HANDLER_ATTR, True)
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(_UtcFormatter(_LEGACY_FORMAT))
    return logger
