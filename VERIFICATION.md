# Verification record

## Completed locally

- 52 Python unit/mock tests passed.
- Python compilation checks passed for `spark_pair.py` and `welcome.py`.
- `bash -n` passed for every shell entrypoint.
- Entrypoint help, argument forwarding and banner-only preview passed.
- Existing-cluster tests cover exact-host scope, matching-port mapping, management/cluster identity, missing addresses, MTU mismatch, failed probes/validation without a final config, and refusal to overwrite.
- New-cluster tests require explicit matching port 0 or 1 before invoking any helper; automatic/mixed-port choices are refused.
- Independent source review compared setup, recipe, image-copy and cache behavior against eugr commit `6430ce19343e5c406a3b9c65d2fc8a9f4bc672ba`.
- NVIDIA recipe checksum was checked against its pinned upstream download.
- Independent OpenSSH configuration check confirmed that the wrapper's first `StrictHostKeyChecking=yes` takes precedence over later upstream `no` options.
- Public deliverable uses placeholders for site configuration and documentation-only test IPs; it contains no actual host addresses, usernames, credentials, caches or journals.

## Not performed

No actual cluster was modified. No sudo/Netplan changes, SSH-key creation, real two-node SSH tests, Docker image/model downloads, GPU inference, RDMA throughput checks, live health checks, or live restoration were performed. Optional uv bootstrap was not executed here. GitHub CI is supplied but was not run on GitHub during local preparation.

The tests cover wrapper behavior, not all upstream software. The README's source-review claims are distinct from hardware validation. A successful real deployment still requires verifying health and a chat completion on the intended pair, checking network isolation and memory use, and testing recovery during an approved maintenance window.
