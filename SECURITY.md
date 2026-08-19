# Security Policy

## Reporting A Vulnerability

Do not disclose suspected vulnerabilities in a public issue, discussion, pull
request, log, or example artifact.

Use GitHub private vulnerability reporting from this repository's **Security**
tab when it is available. If private reporting is unavailable, open a public
issue containing only a request for a private maintainer contact. Do not include
exploit details, sensitive paths, credentials, images, or personal data in that
issue.

Include the following in the private report when possible:

- Affected revision and platform.
- A minimal reproduction or proof of concept.
- Expected and observed impact.
- Whether external models, datasets, galleries, or crafted files are required.
- Any suggested mitigation.

Maintainers will evaluate reports on a best-effort basis. No response or fix
timeline is guaranteed. Security fixes are not promised for historical
releases or unsupported platforms.

## Security Scope

Security-relevant areas include strict JSON and manifest parsing, model and
gallery artifact integrity, path and symlink handling, unsafe deserialization,
dependency supply chain, denial of service, and disclosure of identity or image
data.

Biometric accuracy, research novelty, and a model's ordinary misidentification
rate are not software vulnerabilities by themselves. A bypass of a documented
validation boundary, integrity check, authorization layer, or privacy control
may be security-relevant.

## Deployment Warning

Animal Visual Identification is not a production identity service. It does not provide authentication,
authorization, transport security, encryption at rest, tenancy isolation,
retention enforcement, or privacy consent workflows. The public runtime returns
closed-set candidates and does not provide calibrated unknown rejection.

Treat images, identity mappings, galleries, manifests, receipts, and model files
as sensitive or untrusted according to their source. Run acquisition and model
conversion tools with least privilege, isolate network-enabled downloads, pin
reviewed artifacts, and never expose a raw checkout directly as a service.

See [Known Limitations](docs/KNOWN_LIMITATIONS.md) for non-security capability
boundaries.
