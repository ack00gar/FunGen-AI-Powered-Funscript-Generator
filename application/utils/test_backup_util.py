import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tempfile
import shutil
import stat
import pytest
import logging
from unittest import mock
import platform
from application.utils import backup_util
import builtins

class DummyLogger(logging.Logger):
    def __init__(self):
        super().__init__("dummy")
        self.infos = []
        self.errors = []
        self.criticals = []
    def info(self, msg, *a, **k):
        self.infos.append(msg)
    def error(self, msg, *a, **k):
        self.errors.append(msg)
    def critical(self, msg, *a, **k):
        self.criticals.append(msg)

@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp()
    os.write(fd, b"test data")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_successful_backup(temp_file):
    logger = DummyLogger()
    backup_path = backup_util.backup_file(temp_file, logger=logger)
    assert backup_path is not None
    assert os.path.exists(backup_path)
    with open(backup_path, 'rb') as f:
        assert f.read() == b"test data"
    assert any("Created backup" in msg for msg in logger.infos)
    os.remove(backup_path)

def test_file_not_found():
    logger = DummyLogger()
    backup_path = backup_util.backup_file("/nonexistent/file.txt", logger=logger)
    assert backup_path is None
    assert any("Source file does not exist" in msg for msg in logger.errors)

def test_unique_naming(temp_file):
    logger = DummyLogger()
    backup1 = backup_util.backup_file(temp_file, logger=logger)
    backup2 = backup_util.backup_file(temp_file, logger=logger)
    assert backup1 != backup2
    if backup1:
        assert os.path.exists(backup1)
        os.remove(backup1)
    if backup2:
        assert os.path.exists(backup2)
        os.remove(backup2)

def test_permission_error(temp_file, temp_dir):
    logger = DummyLogger()
    if platform.system() == "Windows":
        # Simulate permission error by mocking os.open
        with mock.patch("os.open", side_effect=PermissionError("denied")):
            backup_path = backup_util.backup_file(temp_file, backup_dir=temp_dir, max_attempts=1, logger=logger)
            assert backup_path is None
            assert any("Failed to create backup" in msg for msg in logger.errors) or any("critical" in msg.lower() for msg in logger.criticals)
    else:
        # On Unix, actually make the dir non-writable
        os.chmod(temp_dir, 0o400)
        try:
            backup_path = backup_util.backup_file(temp_file, backup_dir=temp_dir, max_attempts=1, logger=logger)
            assert backup_path is None
            assert any("Failed to create backup" in msg for msg in logger.errors) or any("critical" in msg.lower() for msg in logger.criticals)
        finally:
            os.chmod(temp_dir, 0o700)

def test_retries_and_fallback(temp_file):
    logger = DummyLogger()
    # Simulate always failing by patching os.open
    with mock.patch("os.open", side_effect=OSError("fail")):
        called = {}
        def fallback(path):
            called['called'] = True
            # Simulate user cancel
            return None
        backup_path = backup_util.backup_file(temp_file, max_attempts=2, logger=logger, on_failure_prompt=fallback)
        assert backup_path is None
        assert called['called']
        assert any("Failed to create backup" in msg for msg in logger.errors) or any("critical" in msg.lower() for msg in logger.criticals)

def test_fallback_success(temp_file, temp_dir):
    logger = DummyLogger()
    fallback_path = os.path.join(temp_dir, "fallback.bak")
    dummy_fd = 100  # Arbitrary dummy file descriptor
    # Patch os.open: fail twice, then succeed for fallback
    def open_side_effect(path, flags, *args, **kwargs):
        if path != fallback_path:
            raise OSError("fail")
        return dummy_fd
    with mock.patch("os.open", side_effect=open_side_effect):
        def fallback(path):
            # Actually create the file for fallback so backup_util can write to it
            with open(fallback_path, 'wb') as f:
                pass
            return fallback_path
        backup_path = backup_util.backup_file(temp_file, max_attempts=2, logger=logger, on_failure_prompt=fallback)
        assert backup_path == fallback_path
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)

def test_batch_backup(temp_file, temp_dir):
    logger = DummyLogger()
    # Create a second file
    file2 = os.path.join(temp_dir, "file2.txt")
    with open(file2, 'wb') as f:
        f.write(b"data2")
    results = backup_util.backup_files_with_retries([temp_file, file2], logger=logger)
    for v in results.values():
        if v:
            assert os.path.exists(v)
            os.remove(v)

def test_batch_some_fail(temp_file, temp_dir):
    logger = DummyLogger()
    missing = os.path.join(temp_dir, "missing.txt")
    results = backup_util.backup_files_with_retries([temp_file, missing], logger=logger)
    assert results[temp_file] is not None
    tempfile_result = results[temp_file]
    if isinstance(tempfile_result, str):
        os.remove(tempfile_result)
    assert results[missing] is None

def test_non_ascii_filename(temp_dir):
    logger = DummyLogger()
    filename = os.path.join(temp_dir, "тестовый_файл.txt")
    with open(filename, 'wb') as f:
        f.write(b"data")
    backup_path = backup_util.backup_file(filename, logger=logger)
    assert backup_path is not None
    assert os.path.exists(backup_path)
    os.remove(backup_path)
    os.remove(filename)

def test_long_filename(temp_dir):
    logger = DummyLogger()
    filename = os.path.join(temp_dir, "a" * 100 + ".txt")
    with open(filename, 'wb') as f:
        f.write(b"data")
    backup_path = backup_util.backup_file(filename, logger=logger)
    assert backup_path is not None
    assert os.path.exists(backup_path)
    os.remove(backup_path)
    os.remove(filename) 