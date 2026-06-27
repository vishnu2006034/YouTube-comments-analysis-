# TODO

- [x] Fix Gemini API key handling in `src/analyzer_local.py`.
  - Read `GEMINI_API_KEY` from environment.
  - ~~Keep existing hardcoded key as fallback.~~ Removed — keys must come from `.env`.
  - Improve error message / propagate underlying Gemini errors.
- [x] Remove all hardcoded API keys from source code.
- [x] Fix `.gitignore` to exclude generated files, databases, and bytecache.
- [x] Fix `requirements.txt` with all actual dependencies.
- [x] Archive legacy `data.py` to `_legacy/`.
- [x] Fix `JobStatus` dataclass (`created_at` mutable default, missing `analysis_path` field).
- [x] Fix progress bar regression (jumps from 90% back to 40%).
- [x] Add `src/__init__.py` for proper package imports.
- [x] Use `RotatingFileHandler` for log rotation.
- [x] Remove redundant timestamp regex pattern.
- [x] Update README to match actual project structure.
- [ ] Verify Gemini quota/billing; current run may fail with HTTP 429 quota exceeded.
- [ ] Add CSRF protection to POST endpoints.
- [ ] Consider adding rate limiting to the `/analyze` endpoint.
