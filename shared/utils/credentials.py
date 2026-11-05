# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Secrets in the OS credential manager (Windows Credential Manager, macOS Keychain, GNOME Keyring through the keyring package), beside the
notification destinations: wgp_config.json keeps only a credential id."""
from shared.notifications.secure_store import SecureStorageError, _keyring, availability_error

SERVICE_NAME = "WanGP Credentials"

__all__ = ["SecureStorageError", "availability_error", "delete_secret", "load_secret", "save_secret"]


def load_secret(credential_id: str) -> str:
    try:
        return _keyring().get_password(SERVICE_NAME, credential_id) or ""
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not read a credential from OS credential storage ({type(error).__name__}).") from error


def save_secret(credential_id: str, value: str) -> None:
    try:
        _keyring().set_password(SERVICE_NAME, credential_id, value)
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not save a credential in OS credential storage ({type(error).__name__}).") from error


def delete_secret(credential_id: str) -> None:
    try:
        keyring = _keyring()
        if keyring.get_password(SERVICE_NAME, credential_id) is not None:
            keyring.delete_password(SERVICE_NAME, credential_id)
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not remove a credential from OS credential storage ({type(error).__name__}).") from error
