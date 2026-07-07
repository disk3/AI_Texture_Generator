"""Tiny requests-compatible fallback built on urllib.

This is intentionally small and only implements the subset used by API image
generation clients. It lets the API generation path work when artists have not
installed requests.
"""

from __future__ import annotations

import json as jsonlib
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid


class HTTPError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"HTTP {response.status_code}: {response.text[:300]}")


class ReadTimeout(TimeoutError):
    pass


class _Exceptions:
    HTTPError = HTTPError
    ReadTimeout = ReadTimeout
    Timeout = TimeoutError
    ConnectionError = OSError


exceptions = _Exceptions()


class Response:
    def __init__(self, status_code: int, headers: dict, content: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = content

    @property
    def text(self) -> str:
        charset = "utf-8"
        content_type = self.headers.get("Content-Type") or self.headers.get("content-type") or ""
        if "charset=" in content_type:
            charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip()
        return self.content.decode(charset or "utf-8", errors="replace")

    def json(self):
        return jsonlib.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(self)


def _timeout_value(timeout):
    if isinstance(timeout, (tuple, list)) and timeout:
        return float(timeout[-1])
    if timeout is None:
        return None
    return float(timeout)


def _request(method: str, url: str, headers=None, body: bytes | None = None, timeout=None) -> Response:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_timeout_value(timeout)) as resp:
            return Response(resp.getcode(), dict(resp.headers), resp.read())
    except urllib.error.HTTPError as e:
        return Response(e.code, dict(e.headers), e.read())
    except socket.timeout as e:
        raise ReadTimeout(str(e)) from e


def get(url: str, headers=None, timeout=None):
    return _request("GET", url, headers=headers, timeout=timeout)


def post(url: str, headers=None, json=None, data=None, files=None, timeout=None):
    headers = dict(headers or {})
    body = b""
    if files:
        boundary = f"----AITexBoundary{uuid.uuid4().hex}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        chunks = []
        for key, value in (data or {}).items():
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            chunks.append(str(value).encode())
            chunks.append(b"\r\n")
        for key, file_info in files.items():
            filename, payload, mime = file_info
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
            )
            chunks.append(f"Content-Type: {mime or 'application/octet-stream'}\r\n\r\n".encode())
            chunks.append(payload)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
    elif json is not None:
        body = jsonlib.dumps(json).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif isinstance(data, dict):
        body = urllib.parse.urlencode(data).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif data is not None:
        body = data if isinstance(data, bytes) else str(data).encode("utf-8")
    return _request("POST", url, headers=headers, body=body, timeout=timeout)
