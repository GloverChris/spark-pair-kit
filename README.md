# Spark Pair Kit

**Two 128GB DGX Sparks. One guided GLM 5.3 Flash setup.**

An independent, reviewable helper around the community launcher used in NVIDIA's current DGX Spark playbooks. Run it on the **head Spark**, not your laptop. It can keep an existing NVIDIA Sync cluster or guide a fresh, directly cabled pair through cluster setup, downloads, synchronization, and serving.

```text
       .---[ S P A R K ]===========[ S P A R K ]---.
       |          128 GB    +    128 GB           |
       '-------------------+---------------------'
                           |
                    .------v------.
                    |  GLM  5.3   |
                    |    FLASH    |
                    '-------------'
           TWO SPARKS. ONE GUIDED LAUNCH.
```

**Status:** source-reviewed and tested with offline mocks. **Not end-to-end tested on a physical pair.** This is not an NVIDIA product, a production security boundary, a guaranteed performance recipe, or a promise of unattended success.

## Start here

1. Download/clone this repository onto the head Spark and open a terminal in its folder.
2. Read the security notes below, then run:

   ```bash
   bash setup.sh
   ```

3. Answer the short prompts. The wizard explains and asks before applying network/access changes or running the privileged model containers. Existing cluster users choose **already clustered** and skip network setup.

The entrypoint displays the welcome art, collects the pair's addresses, offers to install missing head-side helper tools, downloads pinned source files, previews optional clustering, validates the selected pair, synchronizes the image and model, and starts the server. It saves private progress so it can be rerun after a fix.

This is **one terminal entrypoint**, not a portable double-click installer. Firmware, DGX OS, Docker, cabling, site network approval, host-key verification and occasional re-login cannot safely be skipped. The wizard stops with an actionable error instead of guessing or changing those settings silently. Do not run it with `sudo`; approved operations request sudo only when needed.

Preview just the artwork, without inspecting or changing your machine:

```bash
bash setup.sh --banner-only
```

## Requirements

- Exactly **two DGX Spark / GB10 systems with 128GB unified memory each**, ARM64 DGX OS. A 64GB pair, mixed memory pair, other GPUs, switch fabric, ring or more nodes is outside this kit.
- A direct, supported QSFP cable connected to the **same numbered physical port** on both Sparks. Follow NVIDIA's cabling guide. Two logical network interfaces per physical port are expected; do not assume they are two separate cables.
- Management LAN reachable on both devices, same normal login username and home path on both, SSH and sudo access, and internet access for downloads.
- Working NVIDIA drivers, Docker and NVIDIA Container Toolkit on both. Docker must use the local system socket (`unix:///var/run/docker.sock`), without DOCKER_HOST/DOCKER_CONTEXT/TLS overrides; remote or rootless daemon targets are refused. The kit does not install/reconfigure these or change firewall rules.
- On both: Python 3.10+, `docker`, `nvidia-smi`, `rsync`, `ip`, `ping`, OpenSSH and the standard DGX OS tooling. Manual clustering also needs Ubuntu Netplan with Python `yaml`, `runuser`, `ssh-keygen` and group tools, per upstream.
- Plan for **at least 300 GiB free on each cache volume**, plus Docker storage requirements and free space for a temporary `docker save` archive, normally under `/tmp`. This configurable floor is a planning heuristic, not a measured maximum. Image extraction, model revisions and compilation can require more. No automatic cleanup is performed.
- Stop other GPU workloads yourself before launch. The kit refuses known active GPU containers/processes and never automatically stops Qwen or other services. Two Spark memories do not become a general-purpose single 256GB RAM pool.

## What the wizard changes

| Stage | Changes |
| --- | --- |
| Helper tools | If approved: Ubuntu metadata/package installation for missing head tools; isolated `uv==0.9.7` installation from PyPI when uv is absent. No driver or Docker install. Existing uv is reused. |
| Source preparation | Dedicated private workspace, pinned eugr Git checkout and checksum-verified NVIDIA recipe. No `curl | sh`. |
| Existing cluster | You supply the verified head/worker CX7 IPs and matching physical port. Read-only checks contact only the specified worker management and cluster addresses, validate both rails and node identity, then atomically save **this kit's** `cluster.env`. No subnet scan or network reconfiguration. |
| New cluster, preview | Upstream SSH/sudo authentication and temporary validation. No intended persistent network/access changes. |
| New cluster, apply | Explicit approval only: Netplan changes/application, dedicated persistent SSH keys and trust/authorized_keys entries, Docker group changes and saved recovery journals on both nodes. |
| Model setup | Pull/record image identity, explicit image copy, incremental model download and worker synchronization into an isolated HF cache, privileged containers on both nodes. |
| Stop | Only the configured GLM container name, verified on each node. Auto-remove containers lose their writable layer; bind-mounted caches remain. |

The terminal stays attached to foreground serving. In a second head terminal run `health`. A container saying “running” is not enough. If you close the launch terminal or press Ctrl-C, upstream cleanup is best-effort; verify both nodes with `status` and use `stop` if needed.

## Defaults chosen for sharing

- **No machine names, addresses, usernames or tokens** in the sample config. Each person supplies their own.
- Separate workspace: `~/.local/share/spark-pair-kit` with mode 700.
- Separate model cache: `WORK_DIR/hf-cache`, private on both nodes. Symlinked cache/work paths are rejected. This avoids intentionally editing an existing Qwen/Hugging Face cache, at the cost of extra storage. Both nodes must use the same home path. `--no-cache-dirs` disables upstream mounts of shared compiler/runtime caches; repeated starts may recompile more work.
- Separate container name: `spark_pair_glm53`. Existing containers of that name cause refusal, rather than being reused.
- API: **127.0.0.1:8001**, accessible through an SSH tunnel. This differs from NVIDIA's all-interface port-8000 example.
- Native multi-node vLLM, no Ray. Tensor parallel 2 comes from NVIDIA's YAML.
- Initial context: **32,768 tokens**, rather than the official recipe's 500,000 ceiling. This is a conservative starting configuration, not a tested memory guarantee. The official memory utilization and other backend flags remain unchanged.
- No auto-start service, daemon, cloud exposure, firewall opening, port forwarding on routers, or system-wide cache deletion.

Loopback API binding is consistent with the reviewed upstream HTTP/multiprocessing code: distributed rendezvous and per-node `VLLM_HOST_IP` remain on cluster addresses. It still requires a real two-node smoke test before being called verified on your hardware.

## Manual / advanced usage

The wizard is the easiest path. For explicit stages:

```bash
cp settings.example.json settings.local.json
chmod 600 settings.local.json
# Edit the two management IPv4 fields. Keep secrets out of this file.
bash spark-pair.sh fetch
```

### A. Already clustered with NVIDIA Sync

Read the verified **primary CX7 IPv4 addresses** from NVIDIA Sync or your existing cluster configuration. Fill `head_cluster_ip`, `worker_cluster_ip`, and the matching physical `ports` value (`"0"` or `"1"`) in your private settings. These are operator inputs, not automatically discovered guesses. Verify SSH host fingerprints for the worker's management and cluster addresses first.

```bash
bash spark-pair.sh configure-existing
bash spark-pair.sh check-config
```

`configure-existing` reads local network state and contacts only the **two exact worker addresses you supplied**. It does not enumerate or scan a subnet, broadcast for candidates, or accept new host keys. It verifies that the worker's management and cluster addresses identify the same node, both management/primary cluster addresses belong to their intended nodes, both selected twin interfaces/HCA names exist, rail subnets match, and peer MTUs agree. It writes `cluster.env` atomically only after every check passes. A failed probe/validation leaves no final configuration file.

An existing `cluster.env` is never overwritten. Inspect it and make deliberate local edits if necessary; do not rediscover or reconfigure a working cluster automatically.

### B. Fresh pair needing clustering

First fill `subnet_pool` with a **site-approved unused RFC1918 CIDR** containing at least two /24s, and explicitly set `ports` to `"0"` or `"1"` after checking that the cable uses that **same physical port number on both devices**. Automatic selection and cross-port pairs are refused before the helper runs, including on preview. Management and CX7 addresses are different roles. Never copy somebody else's addresses. The helper checks local route conflicts but cannot discover every network used elsewhere in your organization or VPN.

```bash
bash cluster-pair.sh plan
# Read the plan, verify fingerprints independently, arrange recovery access.
bash cluster-pair.sh apply --ack-network-changes
```

This calls the pinned upstream `setup-cluster.sh`, with direct topology, an explicitly selected matching physical port, MTU 9000 and dedicated state/config paths. The wrapper asks for `APPLY`; upstream retains its own plan confirmation. No `--yes` is sent to the network helper.

Afterward, inspect `WORK_DIR/cluster.env`, fill the two cluster addresses into settings, and start a fresh login if Docker group membership was added. **Do not apply this path over an already-working NVIDIA Sync setup.**

```bash
bash cluster-pair.sh doctor       # upstream diagnosis only, --dry-run
bash cluster-pair.sh restore --ack-network-changes
```

Restore uses the original head/user, the kit's saved manifest and the root-only journals on every node. Preserve these until restoration is complete. Restore may disconnect you and can fail on unreachable nodes or later conflicting edits. It is not a guaranteed recovery mechanism. Never edit/delete journals casually or copy them into support bundles: they may contain private keys or previously stored data.

### C. Deploy the model

```bash
bash serve-glm.sh preflight
bash serve-glm.sh plan
bash serve-glm.sh setup --ack-runtime-risk
```

`plan` prints wrapper commands without executing upstream code. We deliberately do **not** rely on upstream `--dry-run` as a no-change guarantee, because missing configuration can trigger discovery and write settings first.

`setup` explicitly copies the prepared image, verifies identical image IDs or equivalent runnable content, and forces an **incremental** Hugging Face download/sync. It does not force-delete cached weights. This avoids upstream's normal `--setup` shortcut that can skip copying a model already cached on the head.

After both image and model synchronization finish, the kit writes a private preparation receipt. The wizard offers a no-download restart on subsequent runs. After stopping a successfully prepared instance, restart without downloading:

```bash
bash serve-glm.sh start --ack-runtime-risk
```

`start` checks for the completed preparation receipt and a smaller 20 GiB cache-volume headroom floor, rather than requiring another 300 GiB. These are conservative checks, not precise space predictions. Interrupted preparation still needs `setup` and the larger floor; if storage is tight, free space deliberately rather than bypassing checks.

All stages accept `--config /path/to/settings.local.json` and `--work-dir /absolute/private/path`. Use the same choices on every invocation. Work paths with spaces, symlinks or shell punctuation are deliberately unsupported because of upstream shell assumptions. A private config file may live elsewhere.

`--yes` skips the wrapper's typed confirmation only. It does not grant consent by itself for networking/runtime risks, skip upstream cluster approval, solve host-key prompts, or suppress sudo authentication. Use it only after understanding the exact action.

## Verify and connect

On the head, while launch is running in another terminal:

```bash
bash serve-glm.sh status
bash serve-glm.sh health
bash serve-glm.sh tunnel
```

`health` checks HTTP 200, the expected model ID, and a small chat-completions request. A chat response with choices is only an API smoke test, not a quality or performance benchmark. First requests can be slow because of compilation. Reasoning may consume the short token allowance.

`tunnel` prints an SSH command for your laptop using your configured management address. Replace `LOGIN` with your Spark username and verify its host fingerprint. The client base URL is `http://127.0.0.1:8001/v1`, with model `nvidia/GLM-5.3-Flash-FP4-Local` unless you changed the API port.

The endpoint has no API authentication by default. Loopback access and an authenticated SSH tunnel reduce exposure but do not isolate it from other local users. Add an appropriately secured gateway separately if you need team access; do not bind publicly just to make connection troubleshooting easier.

Stop only this model:

```bash
bash serve-glm.sh stop
```

Stop verification fails if a node cannot be reached. Never infer that both containers stopped from an interrupted command or upstream log line alone.

## Security and reproducibility

1. **Community code and privileged containers:** the launcher is maintained outside NVIDIA. The stock path uses GPU access, `--privileged`, host networking/IPC and bind mounts. A malicious or vulnerable image can compromise the host. Review and approve it under workplace policy. This kit is not suitable as a multi-tenant isolation boundary.
2. **Network/access changes:** the optional cluster helper creates persistent SSH credentials and grants Docker group membership, which is effectively root-level control. Only a responsible operator should approve it. Have independent management/console access and backups.
3. **SSH fingerprints:** the wrapper's own checks use strict host-key checking. Some pinned upstream copy/launch commands explicitly disable it, so runtime commands are run through private `ssh`/`scp` wrappers that put `StrictHostKeyChecking=yes` first. Unknown/changed keys fail rather than being silently accepted. Verify keys independently; never delete known-host entries merely to silence a warning. This protects the reviewed PATH-based commands, not every conceivable future upstream implementation. Existing-pair configuration uses bounded, strict checks and never invokes upstream discovery.
4. **Trusted fabric still required:** HTTP is loopback, but distributed rendezvous/RDMA traffic is not an authenticated public API. Isolate both management and cluster traffic. No security-setting changes are made to make it “work.”
5. **Pins have limits:** launcher commit and NVIDIA recipe commit/checksum are fixed below. The image tag is dated but mutable; first setup records its actual ID/digests and later starts reject changes. This is trust-on-first-pull, not a preverified registry digest. Image identity is checked on both nodes after copying. When Docker image stores expose different IDs, the pinned upstream content-fingerprint helper compares platform, layers and runtime configuration; this is equivalence checking, not an image signature. The official model download is **not revision-pinned**, and upstream's `uvx hf` tool dependencies are not fully locked. This is not a fully reproducible software supply chain.
6. **Credentials and sharing:** never put tokens in settings, shell arguments, `.env`, logs, screenshots or Git commits. If the public model requires authentication, use the Hugging Face CLI in the kit's isolated cache with your own approved account; the cache is mounted inside privileged containers, so that credential is exposed to them. Prefer the public unauthenticated download when possible. Do not authenticate or accept terms for a colleague.
7. **No automatic destructive recovery:** the kit does not prune Docker, remove model files, kill arbitrary processes, reset networking, disable firewalls, or alter SSH daemon policy. Resolve conflicts deliberately.

### Pinned source identities

- eugr launcher: `6430ce19343e5c406a3b9c65d2fc8a9f4bc672ba`
- NVIDIA playbooks: `30883150541743d580c9c347700c6ac99fc396d2`
- Official GLM YAML SHA-256: `063ba087ff51b9f9ca37fdcc3debc0052fe8eb4db3c53497b99a9231f38ba6e3`
- Official recipe image tag: `eugr/spark-vllm-b12x:nightly-20261015`
- Optional isolated bootstrap package: `uv==0.9.7`; runner dependency: `pyyaml==6.0.2`

Review before updating any pin. The wrapper rejects a modified tracked or untracked launcher checkout. Downloads are staged and atomically installed, so a failed Git fetch does not poison the final checkout. Unfinished staging files from an uncatchable process kill may remain private and can be inspected manually. `uv run --no-project` avoids modifying an unrelated Python project. Third-party code is downloaded at runtime, not included in this repository; its original licenses apply. MIT covers only this project's original files.

## Troubleshooting

- **“Run on the configured head”**: use a terminal on that Spark. The first management address must belong to it.
- **Missing package / Docker permission**: install the named prerequisite through your administrator. If membership just changed, log out/in; do not chmod the Docker socket or run this whole kit as root.
- **SSH verification failure**: verify the actual Spark host fingerprint using a trusted route, then establish normal SSH to that exact cluster address. A valid Sync alias does not necessarily establish trust for a raw IP. Rerun after trust/authentication is correct.
- **New setup network failure**: keep management access and the journals. Use `doctor` first; arrange a maintenance window before `restore`. Do not keep applying new pools/topologies over partial state.
- **Model download interrupted or worker incomplete**: rerun `setup`, which deliberately rechecks/incrementally syncs files. `start` alone assumes a completed previous setup. Downloads can take hours on a slow connection.
- **Image tag changed / worker mismatch**: inspect `image-lock.json` and both local image identities. Do not delete the receipt just to suppress a mismatch. Deliberately review an upgrade in a new private workspace if needed.
- **Container name exists**: inspect it. Stop this kit's running instance with `stop`. A leftover stopped container is not automatically deleted; identify it and remove only that container yourself if appropriate.
- **Out of memory / startup fails**: stop other workloads, review both node logs and memory, and keep the 32K initial context. Nothing here guarantees a particular context size or throughput. Do not raise memory utilization blindly.
- **Port occupied**: set unused `api_port` / `master_port` in the private settings; keep them distinct. Changing the API port also changes the tunnel/client URL.
- **Health fails while loading**: wait for the API-start message, then retry. Check both nodes. `status` alone does not test inference or RDMA bandwidth.

## Test / contribution workflow

No GPU required:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile spark_pair.py welcome.py
bash -n setup.sh spark-pair.sh cluster-pair.sh serve-glm.sh
bash setup.sh --banner-only
```

Tests mock orchestration and validate refusal paths, safe arguments, source/config guards, explicit synchronization, strict SSH wrappers welcome/help behavior, explicit port selection, bounded existing-pair configuration, identity/address mapping, and failure-without-overwrite behavior. They do not test Netplan application, sudo, SSH to real devices, image/model downloads, Docker launch, GPU memory, RDMA transport, API readiness or restoration on hardware.

Before contributing: do not commit any private settings, downloaded launcher, cache, journals, logs, host keys, image receipts or credentials. Use only documentation IPs in tests. Report security issues privately to the repository owner, with secrets removed.

## References

- [NVIDIA: serve LLMs across two Sparks, guided recipe path](https://build.nvidia.com/spark/vllm-cluster/intermediate)
- [NVIDIA: cable and configure multiple Sparks](https://build.nvidia.com/spark/connect-multiple-sparks/overview)
- [NVIDIA: manual cluster setup](https://build.nvidia.com/spark/connect-multiple-sparks/advance)
- [NVIDIA: official GLM YAML at the pinned commit](https://github.com/NVIDIA/dgx-spark-playbooks/blob/30883150541743d580c9c347700c6ac99fc396d2/nvidia/playbook-vllm-cluster/assets/glm53-flash-2x128gb.yaml)
- [NVIDIA: detailed local inference recipe](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark-2gpu/index.html?model=z-ai%2Fglm-5.3-flash&os=linux&backend=vllm&memory=128&precision=fp4-local&memoryConfig=128)
- [eugr: pinned setup/restore documentation](https://github.com/eugr/spark-vllm-docker/blob/6430ce19343e5c406a3b9c65d2fc8a9f4bc672ba/docs/NETWORKING.md)
- [eugr: pinned launcher repository](https://github.com/eugr/spark-vllm-docker/tree/6430ce19343e5c406a3b9c65d2fc8a9f4bc672ba)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
