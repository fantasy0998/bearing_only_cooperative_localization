# AGENTS (src)

## Coding Rules
- Treat `src/` as production code: prioritize correctness and reproducibility.
- Keep module interfaces stable unless the user explicitly asks for breaking changes.
- Prefer NumPy vectorization over unnecessary Python loops in hot paths.
- Validate shapes/units at boundaries; fail fast with clear errors.

## Style
- New public functions need type hints and short docstrings.
- Reuse existing helpers before adding new abstractions.
- Keep diffs focused; avoid drive-by refactors.

## Tests
- Run targeted tests for changed modules first, then run full suite if changes are broad.
- If a fix addresses a bug, add/extend a regression test in `test_*.py`.

