from __future__ import annotations

import base64
import hashlib
import hmac
import os


HASH_SCHEME = "pbkdf2_sha256"


def hash_password(password: str, *, salt: str, iterations: int = 600_000) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        iterations,
    )
    return f"{HASH_SCHEME}${iterations}${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iteration_text, salt, expected = encoded.split("$", 3)
        if scheme != HASH_SCHEME:
            return False
        actual = hash_password(password, salt=salt, iterations=int(iteration_text))
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, encoded)


def authenticated(authorization: str | None) -> bool:
    username = os.environ.get("AUTH_USERNAME", "")
    password_hash = os.environ.get("AUTH_PASSWORD_HASH", "")
    if not username or not password_hash or not authorization:
        return False
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "basic":
            return False
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
        supplied_username, supplied_password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    return hmac.compare_digest(supplied_username, username) and verify_password(
        supplied_password, password_hash
    )
