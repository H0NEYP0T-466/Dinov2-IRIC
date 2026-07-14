# 🛡 Security Policy

We take the security of **Dinov2-ISIC** seriously. If you discover a vulnerability, please follow the guidelines below so we can address it responsibly and promptly.

---

## 📋 Table of Contents

- [Supported Versions](#-supported-versions)
- [Reporting a Vulnerability](#-reporting-a-vulnerability)
- [What to Expect](#-what-to-expect)
- [Security Best Practices for Users](#-security-best-practices-for-users)
- [Contact](#-contact)

---

## 🔄 Supported Versions

We actively maintain and provide security updates for the latest release of the `master` branch.

| Version | Supported          |
|---------|--------------------|
| `master` (latest) | ✅ Yes |
| Older commits      | ❌ No  |

Please always test against the latest `master` before reporting.

---

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them responsibly by one of the following methods:

1. **GitHub (preferred):** Use [GitHub Private Vulnerability Reporting](https://github.com/H0NEYP0T-466/Dinov2-ISIC/security/advisories/new) if enabled.
2. **Email:** Send details to the maintainer at **`fa-23-bscs-466@lgu.edu.pk`** with the subject line `[SECURITY] Dinov2-ISIC vulnerability`.

Please include the following in your report:

- A **clear description** of the vulnerability.
- **Steps to reproduce** (or a proof-of-concept) the issue.
- The **impact** / severity you assess (e.g., low / medium / high / critical).
- The **affected component** (backend API, frontend, training script, dependencies).
- Your **version/commit** being tested.
- Any **suggested fix** or mitigation (optional but appreciated).

---

## ⏱ What to Expect

When you report a vulnerability, here's how we handle it:

1. **Acknowledgment** — We'll confirm receipt of your report within **48 hours**.
2. **Investigation** — We'll assess the severity and validity, and may reach out for clarification.
3. **Fix & Test** — Once confirmed, we develop and test a fix.
4. **Disclosure** — We'll coordinate public disclosure with you. We credit reporters who wish to be named.
5. **Closure** — The fix is merged, a security advisory is published (if applicable), and the reporter is thanked.

We aim to resolve valid critical vulnerabilities within **14 days** of confirmation.

---

## 🔒 Security Best Practices for Users

When deploying Dinov2-ISIC (especially the backend), please follow these practices:

- **Do not expose the API publicly** without authentication. The default `CORS_ORIGINS=["*"]` is for development only — restrict it in production.
- **Validate & sanitize** all uploads. The backend processes image files; ensure uploaded content images are checked/limited in size.
- **Use environment variables** (`.env`) for sensitive configuration — never hardcode secrets.
- **Keep dependencies up to date** — run `pip audit` / `npm audit` regularly.
- **Run the backend behind a reverse proxy** (e.g., nginx) in production with HTTPS enabled.
- **Resource limits** — the backend loads the model (~86M params) into memory; monitor memory/GPU usage under load.
- **Docker** — if using Docker, run the container as a non-root user and scan images for known vulnerabilities.

---

## 📞 Contact

- **Security advisories:** [GitHub Security Advisories](https://github.com/H0NEYP0T-466/Dinov2-ISIC/security/advisories)
- **Private email:** `fa-23-bscs-466@lgu.edu.pk`
- **General questions:** Open a [GitHub Discussion](../../discussions) or [Issue](../../issues).

---

Thank you for helping keep Dinov2-ISIC and its users safe. 🙏
