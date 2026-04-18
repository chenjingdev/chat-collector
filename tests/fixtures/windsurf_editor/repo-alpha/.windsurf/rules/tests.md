---
trigger: glob
globs:
  - tests/**/*.py
description: Apply stricter assertions in test files.
---

Test files should prefer exact archive-shape assertions over loose substring checks.
