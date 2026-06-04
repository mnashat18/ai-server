import os
import tempfile
import requests
from urllib.parse import urlparse
from config import MAX_DOWNLOAD_BYTES


def is_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def directus_auth_headers(url: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}

    token = os.getenv("DIRECTUS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def download_temp_file(url: str, suffix: str, timeout: tuple[int, int] = (10, 30)):
    headers = directus_auth_headers(url)

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )

    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0

    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            tmp.close()
            os.remove(tmp.name)
            raise ValueError("Downloaded file exceeds size limit")

        tmp.write(chunk)

    tmp.close()
    return tmp.name


def remove_temp_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
