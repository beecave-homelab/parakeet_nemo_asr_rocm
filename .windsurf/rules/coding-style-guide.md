---
trigger: model_decision
description: Enforce PEP8, full-path imports, and Google-style docstrings (including for private functions). All functions must have type hints. Auto-fix formatting, import style, and documentation to keep code clean, typed, and well-documented.
---

# Python Coding Standards Enforcement

You are an AI-agent contributing to a Python project. You must follow the rules outlined below **strictly** when generating, editing, or reviewing any Python code.

---

## 📚 Docstrings & Documentation

- Use **Google Style docstrings** for **all functions, classes, and methods**, including private/internal ones.
- Each docstring must include:
  - **Args**: List all input parameters with their types and concise descriptions.
  - **Returns**: Describe the return value and its type.
  - **Raises** (if applicable): Document any exceptions the function may raise.

---

## ✍️ Type Hints

- All function arguments and return values **must be annotated** with explicit type hints.
  - ✅ `def add(x: int, y: int) -> int:`
  - ❌ `def add(x, y):`

---

## 🧼 Code Style

- Adhere to **PEP 8** standards for all formatting, including:
  - 4-space indentation
  - Line length ≤ 79 characters
  - Snake_case for variables/functions, PascalCase for classes
  - Use `isort` and `black` compatible formatting
- **Never** use wildcard imports (`from module import *`).

---

## 📦 Imports

- Use **absolute (full-path) imports** consistently.
  - ✅ `from my_project.module.submodule import MyClass`
  - ❌ `from .submodule import MyClass` or `import submodule`
- Organize imports in this order with blank lines in between:
  1. Standard library
  2. Third-party packages
  3. Local application imports

---

## 🛠️ Error Handling & Auto-Fixing

- If code violates any of these rules, silently **fix it before outputting**.
- This includes adding missing type hints, converting relative imports to absolute, reformatting code, and rewriting docstrings.

---

Always return Python files that are **PEP8-compliant, type-safe, properly documented, and consistently structured**.
