# Security Policy

## Supported Versions

The public preview is supported on the current `main` branch and the
`v0.1.0-preview` prerelease. Older experiment branches, rejected model
candidates, local caches, and generated artifacts are not security-supported
release surfaces.

## Reporting a Vulnerability

Do not post exploit details, private tokens, private paths, or sensitive logs in
a public issue. Use GitHub private vulnerability reporting if it is available
for this repository. If that is not available, open a short public issue asking
for a private contact path and omit technical details until a private channel is
available.

Useful reports include:

- affected commit, release asset, script, or DOS binary;
- commands needed to reproduce the behavior;
- whether the issue affects the host build tools, generated release package,
  DOS runtime, or documentation;
- any minimal non-sensitive logs.

Model output quality, hallucination, and prompt-response behavior should be
reported as normal bugs unless they expose host data, private artifacts, or a
release-integrity issue.
