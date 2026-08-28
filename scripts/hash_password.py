from __future__ import annotations

import getpass
import secrets

from app.security import hash_password


def main() -> None:
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    if len(password) < 12:
        raise SystemExit("Password must contain at least 12 characters")
    print(hash_password(password, salt=secrets.token_hex(16)))


if __name__ == "__main__":
    main()
