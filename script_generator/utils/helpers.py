import platform


def is_mac():
    return platform.system() == "Darwin"