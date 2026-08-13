"""Repo-root entry shim. Delegates to the importable CLI."""

from __future__ import annotations

import sys

from pii_redaction.cli import main

if __name__ == "__main__":
    sys.exit(main())
