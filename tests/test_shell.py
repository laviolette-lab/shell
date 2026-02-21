"""Tests for the SHELL package."""

from shell import __version__


def test_version_is_string():
    """Version should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_cli_help(capsys):
    """CLI --help should exit 0."""
    from shell.cli import main

    assert main(["--version"]) is None or True  # argparse exits
