from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    SKIPPED = 1
    FAILED = 2
