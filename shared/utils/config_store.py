# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistence for the application's shared configuration (one server process)."""
import atexit
import json
import os
import tempfile
import threading
from copy import deepcopy


config_lock = threading.RLock()
# Text of the last write per file with the stat of the written file. A file
# that still carries that stat holds that text, so an unchanged configuration
# skips the read before the comparison; a file changed by another writer or
# an external edit is read as before.
_written = {}


def _file_stat(filename):
    try:
        stat = os.stat(filename)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def read_config(filename):
    flush_pending_writes(filename)
    # Windows does not allow replacing a file while a normal Python reader holds it open.
    with config_lock:
        with open(filename, encoding="utf-8") as reader:
            return json.load(reader)


def config_snapshot(config):
    with config_lock:
        return deepcopy(config)


def write_config(config, filename):
    """Serialize writers and replace the file only after the complete JSON is ready."""
    with config_lock:
        text = json.dumps(config, indent=4)
        # A pending deferred write of this file holds an older state of the same
        # dictionary; this text supersedes it.
        with _pending_lock:
            _pending.pop(os.path.abspath(filename), None)
        _write_text(filename, text)


# Frequent updates (the last selected model on every switch) are written by a
# timer thread shortly after the change: the callback does not wait for the
# fsync and replacement. A synchronous write or a read of the same file first
# retires the pending write, and the remaining ones are flushed at exit.
_pending = {}
_pending_lock = threading.Lock()
_pending_timer = None
DEFERRED_WRITE_DELAY = 1.0


def write_config_deferred(config, filename):
    global _pending_timer
    with config_lock:
        text = json.dumps(config, indent=4)
    with _pending_lock:
        _pending[os.path.abspath(filename)] = (filename, text)
        if _pending_timer is None:
            _pending_timer = threading.Timer(DEFERRED_WRITE_DELAY, flush_pending_writes)
            _pending_timer.daemon = True
            _pending_timer.start()


def flush_pending_writes(filename=None):
    global _pending_timer
    with _pending_lock:
        if filename is None:
            items = list(_pending.values())
            _pending.clear()
            if _pending_timer is not None:
                _pending_timer.cancel()
                _pending_timer = None
        else:
            item = _pending.pop(os.path.abspath(filename), None)
            items = [item] if item is not None else []
    for pending_filename, text in items:
        with config_lock:
            _write_text(pending_filename, text)


atexit.register(flush_pending_writes)


def _write_text(filename, text):
    with config_lock:
        # Compare under the same lock as replacement. Read the actual file so
        # another writer or an external edit cannot leave a stale cache here.
        key = os.path.abspath(filename)
        written = _written.get(key)
        current = _file_stat(filename)
        if written is not None and current is not None and written[1] == current:
            if written[0] == text:
                return
        elif current is not None:
            with open(filename, encoding="utf-8") as reader:
                if reader.read() == text:
                    _written[key] = (text, current)
                    return
        fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(filename)}.", suffix=".tmp", dir=os.path.dirname(os.path.abspath(filename)))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as writer:
                writer.write(text)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, filename)
            _written[key] = (text, _file_stat(filename))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def update_config(config, filename, changes, *, remove=()):
    """Commit a patch without replacing the shared dictionary or unrelated values."""
    with config_lock:
        updated = dict(config)
        updated.update(changes)
        for key in remove:
            updated.pop(key, None)
        if filename:
            write_config(updated, filename)
        config.update(changes)
        for key in remove:
            config.pop(key, None)
