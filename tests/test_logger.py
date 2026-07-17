import io
import logging
import os
import unittest
from contextlib import contextmanager

import logger as logger_module


def _snapshot_handler(handler: logging.Handler) -> dict:
    return {
        "handler": handler,
        "level": handler.level,
        "formatter": handler.formatter,
        "stream": getattr(handler, "stream", None),
        "owned": getattr(handler, "_ai_server_owned_handler", False),
        "filters": list(handler.filters),
    }


def _restore_handler(snapshot: dict) -> None:
    handler = snapshot["handler"]
    handler.setLevel(snapshot["level"])
    handler.setFormatter(snapshot["formatter"])
    handler.filters[:] = snapshot["filters"]
    if hasattr(handler, "stream"):
        handler.stream = snapshot["stream"]
    if snapshot["owned"]:
        setattr(handler, "_ai_server_owned_handler", True)
    elif hasattr(handler, "_ai_server_owned_handler"):
        delattr(handler, "_ai_server_owned_handler")


def _snapshot_logger_state(name: str) -> dict:
    logger = logging.getLogger(name)
    return {
        "logger": logger,
        "level": logger.level,
        "propagate": logger.propagate,
        "disabled": logger.disabled,
        "handlers": [_snapshot_handler(handler) for handler in list(logger.handlers)],
    }


def _restore_logger_state(snapshot: dict) -> None:
    logger = snapshot["logger"]
    logger.handlers[:] = []
    for handler_snapshot in snapshot["handlers"]:
        _restore_handler(handler_snapshot)
        logger.addHandler(handler_snapshot["handler"])
    logger.setLevel(snapshot["level"])
    logger.propagate = snapshot["propagate"]
    logger.disabled = snapshot["disabled"]


@contextmanager
def isolated_logging(*, names: str | tuple[str, ...] | list[str], log_level: str | None = None):
    if isinstance(names, str):
        target_names = [names]
    else:
        target_names = list(names)
    if "" not in target_names:
        target_names.append("")

    previous_env = os.environ.get("LOG_LEVEL")
    snapshots = []
    seen = set()
    for name in target_names:
        if name in seen:
            continue
        seen.add(name)
        snapshots.append(_snapshot_logger_state(name))

    try:
        if log_level is None:
            os.environ.pop("LOG_LEVEL", None)
        else:
            os.environ["LOG_LEVEL"] = log_level
        yield
    finally:
        for snapshot in reversed(snapshots):
            _restore_logger_state(snapshot)
        if previous_env is None:
            os.environ.pop("LOG_LEVEL", None)
        else:
            os.environ["LOG_LEVEL"] = previous_env


def _owned_handler(logger: logging.Logger) -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, "_ai_server_owned_handler", False):
            return handler
    raise AssertionError("owned handler not found")


class LoggerTests(unittest.TestCase):
    def test_default_name_and_default_info_level(self):
        with isolated_logging(names="ai-server", log_level=None):
            log = logger_module.get_logger()
            self.assertEqual(log.name, "ai-server")
            self.assertEqual(log.level, logging.INFO)
            self.assertFalse(log.propagate)

    def test_level_parsing_is_case_insensitive_and_whitespace_tolerant(self):
        for value, expected in [(" debug ", logging.DEBUG), (" info ", logging.INFO), (" warning ", logging.WARNING)]:
            with self.subTest(value=value):
                with isolated_logging(names="logger-test-level", log_level=value):
                    log = logger_module.get_logger("logger-test-level")
                    self.assertEqual(log.level, expected)

    def test_supported_logging_levels_and_aliases(self):
        cases = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "WARN": logging.WARNING,
            "FATAL": logging.CRITICAL,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                with isolated_logging(names="logger-test-supported-levels", log_level=value):
                    log = logger_module.get_logger("logger-test-supported-levels")
                    self.assertEqual(log.level, expected)

    def test_invalid_log_level_raises_value_error(self):
        with isolated_logging(names="logger-test-invalid-level", log_level="INVALID"):
            with self.assertRaisesRegex(ValueError, "LOG_LEVEL must be one of"):
                logger_module.get_logger("logger-test-invalid-level")

    def test_invalid_logger_name_is_rejected(self):
        with isolated_logging(names="logger-test-name-validation", log_level=None):
            for value in [None, "", "   ", 123]:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "logger name must be a non-empty string"):
                        logger_module.get_logger(value)  # type: ignore[arg-type]

    def test_repeated_get_logger_calls_return_same_logger(self):
        with isolated_logging(names="logger-test-idempotent", log_level="INFO"):
            first = logger_module.get_logger("logger-test-idempotent")
            second = logger_module.get_logger("logger-test-idempotent")
            self.assertIs(first, second)

    def test_repeated_calls_do_not_add_handlers(self):
        with isolated_logging(names="logger-test-handler-count", log_level="INFO"):
            log = logger_module.get_logger("logger-test-handler-count")
            handler_count = len(log.handlers)
            logger_module.get_logger("logger-test-handler-count")
            self.assertEqual(len(log.handlers), handler_count)

    def test_message_is_emitted_only_once(self):
        with isolated_logging(names="logger-test-single-emission", log_level="INFO"):
            log = logger_module.get_logger("logger-test-single-emission")
            stream = io.StringIO()
            handler = _owned_handler(log)
            original_stream = handler.stream
            handler.stream = stream
            try:
                log.info("hello world")
            finally:
                handler.stream = original_stream

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("hello world", lines[0])

    def test_changing_log_level_updates_logger_and_handler_levels(self):
        with isolated_logging(names="logger-test-update-level", log_level=None):
            os.environ["LOG_LEVEL"] = "INFO"
            log = logger_module.get_logger("logger-test-update-level")
            handler = _owned_handler(log)
            first_level = log.level
            first_handler_level = handler.level

            os.environ["LOG_LEVEL"] = "ERROR"
            log_again = logger_module.get_logger("logger-test-update-level")
            handler_again = _owned_handler(log_again)
            second_level = log_again.level
            second_handler_level = handler_again.level

            self.assertIs(log, log_again)
            self.assertEqual(first_level, logging.INFO)
            self.assertEqual(first_handler_level, logging.INFO)
            self.assertEqual(second_level, logging.ERROR)
            self.assertEqual(second_handler_level, logging.ERROR)

    def test_isolated_logging_restores_exact_logger_state(self):
        log_name = "logger-test-restoration"
        original_stream = io.StringIO()
        original_handler = logging.StreamHandler(original_stream)
        original_handler.setLevel(logging.WARNING)
        original_handler.setFormatter(logging.Formatter("original %(message)s"))
        original_handler.addFilter(lambda record: True)
        logger = logging.getLogger(log_name)
        logger.handlers[:] = [original_handler]
        logger.setLevel(logging.ERROR)
        logger.propagate = True
        logger.disabled = True

        with isolated_logging(names=log_name, log_level="INFO"):
            current = logger_module.get_logger(log_name)
            self.assertNotEqual(current.handlers, [original_handler])
            self.assertTrue(any(getattr(handler, "_ai_server_owned_handler", False) for handler in current.handlers))

        self.assertEqual(logger.handlers, [original_handler])
        self.assertEqual(logger.level, logging.ERROR)
        self.assertTrue(logger.propagate)
        self.assertTrue(logger.disabled)
        self.assertFalse(any(getattr(handler, "_ai_server_owned_handler", False) for handler in logger.handlers))
        self.assertEqual(logger.handlers[0].level, logging.WARNING)
        self.assertEqual(getattr(logger.handlers[0].formatter, "_fmt", None), "original %(message)s")
        self.assertEqual(logger.handlers[0].stream, original_stream)
        self.assertEqual(len(logger.handlers[0].filters), 1)

    def test_propagation_is_disabled_to_prevent_root_duplication(self):
        root_stream = io.StringIO()
        root_handler = logging.StreamHandler(root_stream)
        root_logger = logging.getLogger()
        with isolated_logging(names=("logger-test-propagation", ""), log_level="INFO"):
            root_logger.addHandler(root_handler)
            root_logger.setLevel(logging.INFO)
            log = logger_module.get_logger("logger-test-propagation")
            stream = io.StringIO()
            handler = _owned_handler(log)
            original_stream = handler.stream
            handler.stream = stream
            try:
                log.info("propagation check")
            finally:
                handler.stream = original_stream
            self.assertFalse(log.propagate)
            self.assertEqual(len([line for line in stream.getvalue().splitlines() if line.strip()]), 1)
            self.assertEqual(root_stream.getvalue().strip(), "")

    def test_existing_unrelated_handlers_are_not_deleted(self):
        with isolated_logging(names="logger-test-unrelated", log_level="INFO"):
            log = logging.getLogger("logger-test-unrelated")
            unrelated_stream = io.StringIO()
            unrelated_handler = logging.StreamHandler(unrelated_stream)
            log.addHandler(unrelated_handler)
            before = list(log.handlers)
            logger_module.get_logger("logger-test-unrelated")
            self.assertIn(unrelated_handler, before)
            self.assertIn(unrelated_handler, log.handlers)
            self.assertTrue(any(getattr(handler, "_ai_server_owned_handler", False) for handler in log.handlers))

    def test_legacy_owned_handler_is_reused_and_marked(self):
        with isolated_logging(names="logger-test-legacy", log_level="INFO"):
            log = logging.getLogger("logger-test-legacy")
            legacy_handler = logging.StreamHandler()
            legacy_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            log.addHandler(legacy_handler)
            logger_module.get_logger("logger-test-legacy")
            self.assertEqual(len(log.handlers), 1)
            self.assertIs(log.handlers[0], legacy_handler)
            self.assertTrue(getattr(legacy_handler, "_ai_server_owned_handler", False))

    def test_custom_stream_legacy_shape_is_not_adopted(self):
        with isolated_logging(names="logger-test-legacy-custom-stream", log_level="INFO"):
            log = logging.getLogger("logger-test-legacy-custom-stream")
            custom_stream = io.StringIO()
            legacy_like_handler = logging.StreamHandler(custom_stream)
            legacy_like_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            log.addHandler(legacy_like_handler)
            logger_module.get_logger("logger-test-legacy-custom-stream")
            self.assertEqual(len(log.handlers), 2)
            self.assertIn(legacy_like_handler, log.handlers)
            self.assertFalse(getattr(legacy_like_handler, "_ai_server_owned_handler", False))
            self.assertTrue(any(getattr(handler, "_ai_server_owned_handler", False) for handler in log.handlers))

    def test_custom_level_legacy_shape_is_not_adopted(self):
        with isolated_logging(names="logger-test-legacy-custom-level", log_level="INFO"):
            log = logging.getLogger("logger-test-legacy-custom-level")
            legacy_like_handler = logging.StreamHandler()
            legacy_like_handler.setLevel(logging.INFO)
            legacy_like_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            log.addHandler(legacy_like_handler)
            logger_module.get_logger("logger-test-legacy-custom-level")
            self.assertEqual(len(log.handlers), 2)
            self.assertIn(legacy_like_handler, log.handlers)
            self.assertFalse(getattr(legacy_like_handler, "_ai_server_owned_handler", False))
            self.assertTrue(any(getattr(handler, "_ai_server_owned_handler", False) for handler in log.handlers))

    def test_timestamp_format_is_utc(self):
        with isolated_logging(names="logger-test-timestamp", log_level="INFO"):
            log = logger_module.get_logger("logger-test-timestamp")
            stream = io.StringIO()
            handler = _owned_handler(log)
            original_stream = handler.stream
            handler.stream = stream
            try:
                log.info("timestamp check")
            finally:
                handler.stream = original_stream

        line = stream.getvalue().strip()
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z INFO logger-test-timestamp timestamp check$")


if __name__ == "__main__":
    unittest.main()
