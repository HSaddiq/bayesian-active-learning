try:
    from importlib.metadata import (  # type: ignore
        PackageNotFoundError,
        version,
    )
except ModuleNotFoundError:
    # if using Python 3.7, import from the backport
    from importlib_metadata import (  # type: ignore
        PackageNotFoundError,
        version,
    )

try:
    __version__ = version("bal")
except PackageNotFoundError:
    # package is not installed
    pass
