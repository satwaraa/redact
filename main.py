"""
main — repo-root entry shim. Three lines, no logic.

WHAT GOES HERE
    import sys
    from pii_redaction.cli import main
    if __name__ == "__main__": sys.exit(main())

WHY IT IS THIS SMALL
    Everything that was here — argument parsing, wiring, exit codes — belongs in
    pii_redaction/cli.py, where it is importable and testable. A module at the
    repo root is not part of the installed package: pytest can only import it by
    accident of the working directory, and `pip install .` does not ship it.

    Keeping the shim anyway means `python main.py ...` still works for anyone
    who reaches for the obvious thing, while the real entry point is the console
    script declared in pyproject.toml:

        [project.scripts]
        redact = "pii_redaction.cli:main"

    which gives `uv run redact input.docx -o output.docx`.

    sys.exit(main()) is what turns cli.main's returned code into a process exit
    status — that split is why main() is testable without subprocesses.
"""
