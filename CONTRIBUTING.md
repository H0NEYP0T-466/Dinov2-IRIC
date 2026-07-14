# 🤝 Contributing to Dinov2-ISIC

First off, thank you for considering contributing to **Dinov2-ISIC**! Whether it's a bug report, feature request, documentation improvement, or a code contribution — all forms of help are welcome and valued.

This document provides guidelines and steps to help you contribute effectively.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Development Setup](#-development-setup)
- [Code Style & Linting](#-code-style--linting)
- [Commit Guidelines](#-commit-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Community](#-community)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Dinov2-ISIC.git
   cd Dinov2-ISIC
   ```
3. **Set up** the development environment (see [Development Setup](#-development-setup) below).
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

---

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating a bug report, please check the [existing issues](../../issues) to avoid duplicates. When filing an issue:

- Use the **Bug Report** issue template.
- Provide a **clear, descriptive title**.
- Include **steps to reproduce** the problem.
- Describe the **expected** vs **actual** behavior.
- Specify your **environment** (OS, Python/Node version, GPU, etc.).
- Attach **logs** or **screenshots** if applicable.

### ✨ Suggesting Features

Feature requests are welcome! Use the **Feature Request** issue template and include:

- The **problem** you're trying to solve.
- Your **proposed solution**.
- Any **alternatives** you've considered.
- The **scope** of the feature.

### 📝 Improving Documentation

Documentation improvements are just as valuable as code! Feel free to fix typos, clarify instructions, or add missing information.

---

## 🛠 Development Setup

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place the trained model checkpoint:
#   backend/checkpoints/model_best.pth

python -m app.main               # or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server runs at **http://localhost:8000** with Swagger UI at **http://localhost:8000/docs**.

### Frontend (React + TypeScript + Vite)

```bash
npm install
npm run dev                      # http://localhost:5173
```

### Training (Kaggle / Colab)

```bash
cd kaggle
pip install -r train_requirements.txt
python trainCollab.py
```

---

## 🎨 Code Style & Linting

### Python (Backend)

- Follow **PEP 8** conventions.
- Use **type annotations** on all function signatures.
- Keep functions small and focused (< 50 lines).
- Prefer immutable data structures; never mutate inputs in-place.
- Use the `logging` module — no `print()` statements in production code.

We recommend **ruff** for linting/formatting and **black** as a formatter:

```bash
ruff check .
ruff format .
```

### TypeScript / React (Frontend)

- Follow the existing **ESLint** configuration (`eslint.config.js`).
- Use **TypeScript** types — avoid `any`; prefer `unknown` when uncertain.
- Components should use explicit prop types (interfaces or types).
- Run linting before committing:

```bash
npm run lint
```

### General Rules

- Files should stay under **800 lines** — extract utilities into smaller modules.
- Organize code by **feature/domain**, not by type.
- Write **self-documenting** code with clear variable/function names.
- Add **docstrings/comments** only where the "why" isn't obvious.

---

## 📝 Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <optional scope> <description>

<optional body>

<optional footer>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `build`, `revert`

**Examples:**
```
feat(backend): add TTA (Test-Time Augmentation) endpoint
fix(frontend): resolve confidence bar overflow on mobile
docs: update training setup instructions
```

- Use the **present tense** ("add feature" not "added feature").
- Use the **imperative mood** ("move cursor" not "moves cursor").
- Limit the first line to **72 characters** or fewer.

---

## 🔀 Pull Request Process

1. **Update** your branch with the latest `master`:
   ```bash
   git fetch origin
   git rebase origin/master
   ```
2. **Run all tests** and ensure they pass.
3. **Update documentation** if your change affects behavior.
4. Fill out the **pull request template** completely.
5. Link any related issues (e.g., `Fixes #123`).
6. Request a review from a maintainer.

**PR Checklist:**
- [ ] Code follows the style guidelines
- [ ] Tests added/updated for changes
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Self-review completed

---

## 🧪 Testing

- Add **unit tests** for new utility functions and services.
- Add **integration tests** for API endpoints.
- Verify the **end-to-end flow**: backend → frontend → prediction.
- Run tests before submitting a PR.

**Backend quick test:**
```bash
# Start the server, then:
curl -X POST http://localhost:8000/api/v1/predict -F "file=@test_image.jpg"
```

---

## 📚 Documentation

If your contribution changes how the project works:

- Update the **README.md** if needed.
- Update relevant **docstrings** and comments.
- Add or update the appropriate **issue/PR template** if the workflow changed.

---

## 💬 Community

- Be respectful and constructive in all interactions.
- Help newcomers and share knowledge.
- Discuss major changes in an issue before writing code.

---

Thank you for contributing! 🎉
