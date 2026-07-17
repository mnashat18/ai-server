import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import requests

import utils


@contextmanager
def patched_env(**updates):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        iter_exc: Exception | None = None,
        raise_exc: Exception | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self._iter_exc = iter_exc
        self._raise_exc = raise_exc
        self.closed = False
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True
        if self._raise_exc is not None:
            raise self._raise_exc
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def iter_content(self, chunk_size=8192):
        if self._iter_exc is not None:
            raise self._iter_exc
        yield from self._chunks

    def close(self):
        self.closed = True


class UtilsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = os.path.join(os.getcwd(), f"utils-test-{os.getpid()}-{id(self)}")
        os.makedirs(self.tempdir, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    @contextmanager
    def patched_mkstemp(self):
        original = tempfile.mkstemp

        def _mkstemp(*, suffix=""):
            return original(suffix=suffix, dir=self.tempdir)

        with mock.patch.object(utils.tempfile, "mkstemp", side_effect=_mkstemp):
            yield

    @contextmanager
    def directus_env(self, url="https://directus.example.com", token="secret"):
        with patched_env(DIRECTUS_URL=url, DIRECTUS_TOKEN=token):
            yield

    def test_is_url_accepts_valid_http_and_https_urls(self):
        cases = [
            "http://example.com",
            "https://example.com/path?x=1",
            " https://example.com:443/assets/1 ",
            "http://example.com:80/file",
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(utils.is_url(value))

    def test_is_url_rejects_malformed_and_unsupported_values(self):
        cases = [
            None,
            "",
            "   ",
            "ftp://example.com",
            "http://user:pass@example.com",
            "http://example.com:abc",
            "http://example.com/\npath",
            "http://example.com/\tpath",
            123,
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(utils.is_url(value))

    def test_directus_auth_headers_returns_auth_only_for_exact_origin(self):
        with self.directus_env(url=" https://directus.example.com:443 ", token=" bearer-token "):
            headers = utils.directus_auth_headers("https://directus.example.com/assets/1")
        self.assertEqual(headers, {"Authorization": "Bearer bearer-token"})

    def test_directus_auth_headers_rejects_scheme_host_subdomain_and_port_mismatch(self):
        with self.directus_env():
            cases = [
                "http://directus.example.com/assets/1",
                "https://api.directus.example.com/assets/1",
                "https://directus.example.net/assets/1",
                "https://directus.example.com:444/assets/1",
            ]
            for value in cases:
                with self.subTest(value=value):
                    self.assertEqual(utils.directus_auth_headers(value), {})

    def test_directus_auth_headers_rejects_url_none_and_blank_token(self):
        with self.directus_env(token="secret"):
            self.assertEqual(utils.directus_auth_headers(None), {})
        with self.directus_env(token="   "):
            self.assertEqual(utils.directus_auth_headers("https://directus.example.com/assets/1"), {})

    def test_directus_auth_headers_fails_on_malformed_directus_url_without_exposing_token(self):
        with patched_env(DIRECTUS_URL="not-a-url", DIRECTUS_TOKEN="secret"):
            with self.assertRaisesRegex(ValueError, "DIRECTUS_URL"):
                utils.directus_auth_headers("https://directus.example.com/assets/1")

    def test_download_temp_file_rejects_untrusted_urls_before_requests(self):
        with self.directus_env():
            with mock.patch("utils.requests.get") as mock_get:
                with self.assertRaisesRegex(ValueError, "download URL must match DIRECTUS_URL origin"):
                    utils.download_temp_file("https://evil.example.com/file.bin", ".bin")
                mock_get.assert_not_called()

    def test_download_temp_file_same_origin_uses_authorization_header(self):
        response = FakeResponse(headers={"Content-Length": "3"}, chunks=[b"abc"])
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=response) as mock_get:
                    path = utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.addCleanup(utils.remove_temp_file, path)
        self.assertTrue(response.closed)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_get.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"abc")

    def test_download_temp_file_redirects_are_bounded_and_same_origin_only(self):
        redirect1 = FakeResponse(status_code=302, headers={"Location": "/assets/2"})
        redirect2 = FakeResponse(status_code=301, headers={"Location": "https://directus.example.com/assets/3"})
        final = FakeResponse(headers={"Content-Length": "2"}, chunks=[b"ok"])
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", side_effect=[redirect1, redirect2, final]) as mock_get:
                    path = utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.addCleanup(utils.remove_temp_file, path)
        self.assertTrue(redirect1.closed)
        self.assertTrue(redirect2.closed)
        self.assertTrue(final.closed)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_get.call_args_list[1].kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(mock_get.call_args_list[2].kwargs["headers"], {"Authorization": "Bearer secret"})
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"ok")

    def test_download_temp_file_rejects_cross_origin_redirect_without_following_it(self):
        redirect = FakeResponse(status_code=302, headers={"Location": "https://evil.example.com/file"})
        with self.directus_env():
            with mock.patch("utils.requests.get", return_value=redirect) as mock_get:
                with self.assertRaisesRegex(ValueError, "redirect target must match DIRECTUS_URL origin"):
                    utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(redirect.closed)
        self.assertEqual(mock_get.call_count, 1)

    def test_download_temp_file_rejects_redirect_loops_after_limit(self):
        responses = [
            FakeResponse(status_code=302, headers={"Location": "/assets/loop"}),
            FakeResponse(status_code=302, headers={"Location": "/assets/loop"}),
            FakeResponse(status_code=302, headers={"Location": "/assets/loop"}),
            FakeResponse(status_code=302, headers={"Location": "/assets/loop"}),
        ]
        with self.directus_env():
            with mock.patch("utils.requests.get", side_effect=responses) as mock_get:
                with self.assertRaisesRegex(ValueError, "too many redirects"):
                    utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertEqual(mock_get.call_count, 4)
        self.assertTrue(all(response.closed for response in responses))

    def test_download_temp_file_rejects_content_length_above_limit(self):
        response = FakeResponse(headers={"Content-Length": "100"}, chunks=[b"abc"])
        with self.directus_env():
            with mock.patch.object(utils, "MAX_DOWNLOAD_BYTES", 10):
                with mock.patch("utils.requests.get", return_value=response) as mock_get:
                    with self.assertRaisesRegex(ValueError, "Downloaded file exceeds size limit"):
                        utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(response.closed)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(os.listdir(self.tempdir), [])

    def test_download_temp_file_removes_partial_file_on_streaming_limit_exceeded(self):
        response = FakeResponse(chunks=[b"abc", b"def"])
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch.object(utils, "MAX_DOWNLOAD_BYTES", 5):
                    with mock.patch("utils.requests.get", return_value=response):
                        with self.assertRaisesRegex(ValueError, "Downloaded file exceeds size limit"):
                            utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(response.closed)
        self.assertEqual(os.listdir(self.tempdir), [])

    def test_download_temp_file_removes_partial_file_on_os_fdopen_failure(self):
        response = FakeResponse(chunks=[b"abc"])
        created = {}
        original_mkstemp = tempfile.mkstemp

        def _mkstemp(*, suffix="", dir=None):
            fd, path = original_mkstemp(suffix=suffix, dir=self.tempdir)
            created["fd"] = fd
            created["path"] = path
            return fd, path

        with self.directus_env():
            with mock.patch.object(utils.tempfile, "mkstemp", side_effect=_mkstemp):
                with mock.patch("utils.requests.get", return_value=response):
                    with mock.patch.object(utils.os, "fdopen", side_effect=OSError("fdopen failed")):
                        with mock.patch.object(utils.os, "close", wraps=utils.os.close) as mock_close:
                            with self.assertRaises(OSError):
                                utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
                            mock_close.assert_called_once_with(created["fd"])
        self.assertTrue(response.closed)
        self.assertFalse(os.path.exists(created["path"]))

    def test_download_temp_file_removes_partial_file_on_network_and_write_failure(self):
        stream_error = FakeResponse(iter_exc=OSError("stream failed"))
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=stream_error):
                    with self.assertRaises(OSError):
                        utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(stream_error.closed)
        self.assertEqual(os.listdir(self.tempdir), [])

        class FailingWriter:
            def __init__(self, fd):
                self._handle = original_fdopen(fd, "wb")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self._handle.close()
                return False

            def write(self, chunk):
                raise OSError("write failed")

        write_error = FakeResponse(chunks=[b"abc"])
        original_fdopen = utils.os.fdopen
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=write_error):
                    with mock.patch.object(utils.os, "fdopen", side_effect=lambda fd, mode: FailingWriter(fd)):
                        with self.assertRaises(OSError):
                            utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(write_error.closed)
        self.assertEqual(os.listdir(self.tempdir), [])

    def test_download_temp_file_closes_response_on_success_and_failure(self):
        success_response = FakeResponse(chunks=[b"abc"])
        failure_response = FakeResponse(raise_exc=requests.HTTPError("boom"))
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=success_response):
                    path = utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.addCleanup(utils.remove_temp_file, path)
        self.assertTrue(success_response.closed)

        with self.directus_env():
            with mock.patch("utils.requests.get", return_value=failure_response):
                with self.assertRaises(requests.HTTPError):
                    utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(failure_response.closed)

    def test_download_temp_file_rejects_empty_download_and_cleans_up(self):
        response = FakeResponse(chunks=[])
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=response):
                    with self.assertRaisesRegex(ValueError, "Downloaded file is empty"):
                        utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.assertTrue(response.closed)
        self.assertEqual(os.listdir(self.tempdir), [])

    def test_download_temp_file_returns_real_temporary_path_with_exact_bytes(self):
        response = FakeResponse(headers={"Content-Length": "6"}, chunks=[b"ab", b"cd", b"ef"])
        with self.directus_env():
            with self.patched_mkstemp():
                with mock.patch("utils.requests.get", return_value=response):
                    path = utils.download_temp_file("https://directus.example.com/assets/1", ".bin")
        self.addCleanup(utils.remove_temp_file, path)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"abcdef")

    def test_download_temp_file_rejects_invalid_suffix_and_timeout(self):
        with self.directus_env():
            with mock.patch("utils.requests.get") as mock_get:
                with self.assertRaisesRegex(ValueError, "suffix contains invalid path components"):
                    utils.download_temp_file("https://directus.example.com/assets/1", "../bad")
                for timeout in [
                    (10, -1),
                    (True, 1),
                    ("10", 1),
                    (None, 1),
                    (float("inf"), 1),
                    (1, float("-inf")),
                    (0, 1),
                ]:
                    with self.subTest(timeout=timeout):
                        with self.assertRaisesRegex(ValueError, "timeout must be a 2-item tuple of positive finite numbers"):
                            utils.download_temp_file("https://directus.example.com/assets/1", ".bin", timeout=timeout)  # type: ignore[arg-type]
                mock_get.assert_not_called()

    def test_remove_temp_file_is_idempotent(self):
        path = os.path.join(self.tempdir, "temp-file.bin")
        with open(path, "wb") as handle:
            handle.write(b"x")
        utils.remove_temp_file(path)
        utils.remove_temp_file(path)
        utils.remove_temp_file(os.path.join(self.tempdir, "missing.bin"))
        self.assertFalse(os.path.exists(path))

    def test_clamp01_rejects_bool_nan_and_infinity_and_clamps_valid_numbers(self):
        self.assertEqual(utils.clamp01(True, default=0.25), 0.25)
        self.assertEqual(utils.clamp01(False, default=0.25), 0.25)
        self.assertEqual(utils.clamp01(float("nan"), default=0.25), 0.25)
        self.assertEqual(utils.clamp01(float("inf"), default=0.25), 0.25)
        self.assertEqual(utils.clamp01(-3.0), 0.0)
        self.assertEqual(utils.clamp01(2.0), 1.0)
        self.assertEqual(utils.clamp01(0.42), 0.42)

    def test_safe_number_rejects_bool_nonfinite_and_validates_digits(self):
        for digits in [True, -1, 13]:
            with self.subTest(digits=digits):
                with self.assertRaises(ValueError):
                    utils.safe_number(None, digits=digits)  # type: ignore[arg-type]
        self.assertIsNone(utils.safe_number(None, digits=4))
        self.assertIsNone(utils.safe_number(True))
        self.assertIsNone(utils.safe_number(False))
        self.assertIsNone(utils.safe_number(float("nan")))
        self.assertIsNone(utils.safe_number(float("inf")))
        self.assertEqual(utils.safe_number(1.23456, digits=3), 1.235)
        for digits in [True, -1, 13, "4"]:
            with self.subTest(digits=digits):
                with self.assertRaises(ValueError):
                    utils.safe_number(1.23, digits=digits)  # type: ignore[arg-type]

    def test_clean_warning_codes_preserves_order_and_removes_duplicates(self):
        self.assertEqual(
            utils.clean_warning_codes(["a", "", "a", "b", "  c  ", "b"]),
            ["a", "b", "c"],
        )

    def test_clean_warning_codes_treats_single_string_as_one_value(self):
        self.assertEqual(utils.clean_warning_codes("warning_code"), ["warning_code"])

    def test_sanitize_text_validates_length_and_preserves_fallback(self):
        self.assertEqual(utils.sanitize_text(None, fallback="fallback", max_len=4), "fall")
        self.assertEqual(utils.sanitize_text(" undefined ", fallback="fallback", max_len=20), "fallback")
        self.assertEqual(utils.sanitize_text("abcdef", max_len=3), "abc")
        for max_len in [True, -1, 1.5, "3"]:
            with self.subTest(max_len=max_len):
                with self.assertRaises(ValueError):
                    utils.sanitize_text("value", max_len=max_len)  # type: ignore[arg-type]

    def test_sanitize_payload_preserves_primitives_drops_nonfinite_and_handles_cycles(self):
        cyclic = {}
        cyclic["self"] = cyclic
        payload = utils.sanitize_payload(
            {
                "ok": True,
                "count": 3,
                "ratio": 1.23456789,
                "bad": float("nan"),
                "tuple": (1, float("inf"), "x"),
                "none": None,
                "text": " undefined ",
                "cyclic": cyclic,
                "unsupported": object(),
            }
        )
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["ratio"], 1.234568)
        self.assertEqual(payload["tuple"], [1, "x"])
        self.assertNotIn("bad", payload)
        self.assertNotIn("none", payload)
        self.assertNotIn("text", payload)
        self.assertEqual(payload["cyclic"], {})
        self.assertNotIn("unsupported", payload)
        self.assertIsNone(utils.sanitize_payload(object()))


if __name__ == "__main__":
    unittest.main()
