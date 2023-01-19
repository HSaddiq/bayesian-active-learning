import os
from pathlib import Path

import nox

# don't run format session by default
nox.options.sessions = ["lint", "test"]

# put virtualenv cache on SSD rather than EFS when on platform
if os.environ.get("FACULTY_PROJECT_ID") is not None:
    # add root folder name in case multiple projects are being worked on on a
    # single platform server.
    nox.options.envdir = f"/tmp/nox-cache/{Path(__file__).parent.stem}/"

SOURCES = ["src", "tests", "noxfile.py"]


@nox.session
def test(session):
    "Run unit tests"
    session.install("-r", "requirements.txt")
    # Add other test dependencies here
    session.install("pytest", "pytest-cov")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def lint(session):
    session.install("black", "flake8", "isort", "nb-clean", "nbqa")

    session.run("black", "--check", *SOURCES)
    session.run("isort", "--check", *SOURCES)

    session.run("nbqa", "black", "--check", "notebooks")
    session.run("nbqa", "flake8", "notebooks")

    for nb in Path("notebooks").glob("**/*.ipynb"):
        nb = nb.as_posix()

        if ".ipynb_checkpoints" in nb:
            continue

        session.run("nb-clean", "check", nb)


@nox.session(name="format", reuse_venv=True)
def format_(session):
    session.install("black", "flake8", "nb-clean", "nbqa", "isort")

    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)

    session.run("nbqa", "black", "notebooks")
    session.run("nbqa", "isort", "notebooks")

    for nb in Path("notebooks").glob("**/*.ipynb"):
        nb = nb.as_posix()

        if ".ipynb_checkpoints" in nb:
            continue

        session.run("nb-clean", "clean", nb)
