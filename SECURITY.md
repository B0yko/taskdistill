# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's
[security advisory form](https://github.com/B0yko/taskdistill/security/advisories/new), not in a public issue.
Include the version (`taskdistill --version`), the command or request that triggers the problem and what an
attacker gains. You should get a first answer within a week.

## What taskdistill handles, and how

- **API keys.** The teacher key is read from `TASKDISTILL_TEACHER_API_KEY` (or `OPENROUTER_API_KEY`) and is only
  ever sent to the configured teacher base URL. It is never written to disk.
- **Headers are never stored.** The capture proxy forwards the client's `Authorization` header to the upstream
  and stores request and response bodies only. The cascade server never forwards the client's `Authorization`
  header to the teacher; escalations use the teacher key from the environment.
- **Binding beyond localhost.** `capture` and `serve` refuse to listen on anything other than `127.0.0.1`,
  `::1` or `localhost` unless `TASKDISTILL_SERVER_TOKEN` is set. The proxy then requires the token in the
  `X-Taskdistill-Token` header (and strips it before forwarding); the server requires it as a bearer token.
  Neither adds TLS: put them behind a TLS-terminating reverse proxy if traffic leaves the machine.
- **Stored data.** The workspace (`$TASKDISTILL_HOME`, default `./.taskdistill`) holds captured request and
  response bodies in SQLite. Treat it like the logs of the application whose traffic you capture.
- **PII scrub is best-effort.** Curate replaces e-mail addresses, phone numbers, IBANs, card numbers, IPv4
  addresses and US SSNs with typed placeholders using regular expressions with checksums. It does not detect
  names, street addresses or anything else, and it is not an anonymisation guarantee.
- **Recordings.** The replay recordings shipped with the package contain teacher outputs, token usage and
  timings keyed by a SHA-256 request key. They contain no inputs, headers or keys.

## Supported versions

Only the latest release receives fixes.
