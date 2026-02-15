# Testing Rules

## Environment
Always use the growth conda environment:
```bash
~/.conda/envs/growth/bin/python -m pytest tests/ -v
```

## Conventions
- Test files: `tests/test_<module_name>.py`
- Test functions: `test_<what_is_being_tested>`
- Each test file must be independently runnable
- Use `-v` flag for verbose output, `--tb=short` for concise tracebacks

## Curation Tests
- Quality filtering tests: `tests/test_quality_filtering.py`
- Normalization tests: `tests/test_normalization.py`
- Run specific test: `~/.conda/envs/growth/bin/python -m pytest tests/test_quality_filtering.py -v`

## Preprocessing Tests
- Step-level tests should validate input→output transformation
- Config tests should verify `@dataclass` validation (`__post_init__`)
- Mock external dependencies (ANTs, HD-BET, SynthStrip) in unit tests
- Integration tests may require GPU and large test data — mark with `@pytest.mark.slow`

## Module Dependencies
Do NOT write tests for Phase N+1 until Phase N tests pass.
Phase order: 1 (LoRA) → 2 (SDP) → 3 (Encoding) → 4 (Neural ODE)
