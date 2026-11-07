#!/usr/bin/env python3
"""Guarded, two-node wrapper for the NVIDIA-documented community launcher.

This project does not vendor or replace NVIDIA/eugr deployment code. See README.
"""
from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error

LAUNCHER_URL = "https://github.com/eugr/spark-vllm-docker.git"
LAUNCHER_SHA = "6430ce19343e5c406a3b9c65d2fc8a9f4bc672ba"
PLAYBOOK_SHA = "30883150541743d580c9c347700c6ac99fc396d2"
RECIPE_SHA256 = "063ba087ff51b9f9ca37fdcc3debc0052fe8eb4db3c53497b99a9231f38ba6e3"
RECIPE_URL = (f"https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/{PLAYBOOK_SHA}"
              "/nvidia/playbook-vllm-cluster/assets/glm53-flash-2x128gb.yaml")
IMAGE = "eugr/spark-vllm-b12x:nightly-20261015"
MODEL = "nvidia/GLM-5.3-Flash-FP4-Local"
KEYS = {"head_management_ip", "worker_management_ip", "head_cluster_ip", "worker_cluster_ip",
        "subnet_pool", "ports", "api_port", "master_port", "container_name", "max_model_len",
        "minimum_free_gib"}
DEFAULTS = dict(head_management_ip=None, worker_management_ip=None,
                head_cluster_ip=None, worker_cluster_ip=None, subnet_pool=None,
                ports=None, api_port=8001, master_port=29501,
                container_name="spark_pair_glm53", max_model_len=32768, minimum_free_gib=300)
ENV_KEYS = {"CLUSTER_NODES", "LOCAL_IP", "ETH_IF", "IB_IF", "COPY_HOSTS", "CLUSTER_LINKS",
            "CONTAINER_NCCL_NET_PLUGIN", "CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING",
            "CONTAINER_NCCL_IB_MERGE_NICS", "CONTAINER_NCCL_ALGO"}

class KitError(Exception):
    pass

def require(condition, message):
    if not condition:
        raise KitError(message)

def ipv4(value):
    require(isinstance(value, str), "Set all required IPv4 addresses in your local settings file.")
    try:
        ip = ipaddress.IPv4Address(value)
    except ValueError as exc:
        raise KitError(f"Invalid IPv4 address: {value!r}") from exc
    require(not (ip.is_loopback or ip.is_multicast or ip.is_unspecified or ip.is_link_local),
            "Use a real node's unicast management/cluster IPv4 address, not loopback/link-local.")
    return str(ip)

def integer(value, lo, hi, name):
    require(type(value) is int and lo <= value <= hi, f"{name} must be an integer in {lo}..{hi}.")
    return value

def load_settings(path):
    try:
        user = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise KitError(f"Cannot read settings {path}: {exc}. Copy settings.example.json first.") from exc
    require(isinstance(user, dict) and not set(user) - KEYS, "Unknown settings key, or settings is not an object.")
    cfg = DEFAULTS | user
    for key in ("head_management_ip", "worker_management_ip"):
        cfg[key] = ipv4(cfg[key])
    require(cfg["head_management_ip"] != cfg["worker_management_ip"], "The two management IPs must differ.")
    for key in ("head_cluster_ip", "worker_cluster_ip"):
        if cfg[key] is not None:
            cfg[key] = ipv4(cfg[key])
    require(cfg["ports"] in (None, "0", "1"), "ports must be the explicitly verified matching physical port: 0 or 1.")
    require(isinstance(cfg["container_name"], str) and re.fullmatch(r"spark_pair_[A-Za-z0-9_-]{1,40}", cfg["container_name"]),
            "container_name must start spark_pair_ and contain only letters, numbers, _ or -.")
    for key in ("api_port", "master_port"):
        integer(cfg[key], 1024, 65535, key)
    require(cfg["api_port"] != cfg["master_port"], "API and master ports must differ.")
    integer(cfg["max_model_len"], 1024, 500000, "max_model_len")
    integer(cfg["minimum_free_gib"], 200, 1000000, "minimum_free_gib")
    return cfg

def strict_ssh_environment(work, env):
    """SSH uses first value for each option: put strict checks ahead of upstream flags."""
    bindir = work / "strict-ssh"
    bindir.mkdir(mode=0o700, exist_ok=True)
    require(not bindir.is_symlink(), "SSH wrapper directory must not be a symlink.")
    for name in ("ssh", "scp"):
        original = shutil.which(name)
        require(original is not None, f"Missing {name}.")
        target = bindir / name
        require(not target.is_symlink(), "SSH wrapper must not be a symlink.")
        content = "#!/bin/sh\nexec " + shlex.quote(original) + ' -o StrictHostKeyChecking=yes "$@"\n'
        if target.exists():
            require(target.read_text() == content, "SSH wrapper changed; review it before proceeding.")
        else:
            private_write(target, content)
            target.chmod(0o700)
    env = env.copy()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    return env

def child_environment(work):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("DOTENV_", "CONTAINER_", "VLLM_SPARK_", "BASH_FUNC_")) or key in {
            "BASH_ENV", "ENV", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
            "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "FORCE_DISCOVER", "RSYNC_RSH"}:
            env.pop(key, None)
    env["HF_HOME"] = str(work / "hf-cache")
    return env

def run(argv, *, cwd=None, env=None, capture=False, timeout=None, input_text=None):
    # Never use shell=True. Credential values are never interpolated or printed here.
    print("+ " + shlex.join(map(str, argv)), flush=True)
    try:
        result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True,
                                text=True, capture_output=capture, timeout=timeout, input=input_text)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # Captured diagnostics are necessary to distinguish SSH trust/auth failures.
        # Never print stdout/inspect metadata, and redact common credential forms.
        detail = getattr(exc, "stderr", "") or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        detail = "\n".join(detail.splitlines()[-15:])[-2400:]
        detail = re.sub(r"hf_[A-Za-z0-9]+", "[redacted-token]", detail)
        detail = re.sub(r"(?i)(Bearer\s+|(?:token|password|secret|api[_-]?key)\s*[:=]\s*)\S+", r"\1[redacted]", detail)
        detail = re.sub(r"-----BEGIN .*?PRIVATE KEY-----[\s\S]*", "[redacted-private-key]", detail)
        status = getattr(exc, "returncode", None)
        suffix = f" (exit {status})" if status is not None else f" ({type(exc).__name__})"
        raise KitError(f"Command failed: {argv[0]}{suffix}." + ("\n" + detail if detail else "")) from exc
    return result.stdout.strip() if capture else ""

def private_write(path, text):
    """Atomic, no-overwrite installation; a killed write cannot leave a partial target."""
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite {path}.")
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)  # atomic and fails if another writer made the target
    finally:
        temporary.unlink(missing_ok=True)

def ensure_work(work):
    require(re.fullmatch(r"/[A-Za-z0-9_./-]+", str(work)) is not None,
            "Work path must be absolute, without spaces/shell punctuation (upstream limitation).")
    for path in [work, *work.parents]:
        require(not path.is_symlink(), "Symlinked work-directory paths are not supported.")
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(work.stat().st_uid == os.getuid(), "Work directory must belong to your login user.")
    require(work.stat().st_mode & 0o077 == 0, "Work directory must be private: chmod 700 it yourself.")

def verify_checkout(work):
    repo = work / "launcher"
    require(repo.is_dir(), "Run fetch first.")
    require(run(["git", "rev-parse", "HEAD"], cwd=repo, capture=True) == LAUNCHER_SHA,
            "Launcher commit changed. Use the reviewed pinned checkout.")
    require(not run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, capture=True),
            "Launcher checkout is modified or has untracked files; review in a separate directory.")
    recipe = work / "glm53-flash-2x128gb.yaml"
    require(recipe.is_file() and hashlib.sha256(recipe.read_bytes()).hexdigest() == RECIPE_SHA256,
            "Recipe checksum mismatch or missing recipe. Refusing execution.")
    return repo

def fetch(work):
    repo = work / "launcher"
    if not repo.exists():
        # A failed download leaves only an unused staging directory. A subsequent run
        # creates a fresh staging directory, never resets somebody else's checkout.
        staging = Path(tempfile.mkdtemp(prefix=".launcher-download-", dir=work))
        try:
            run(["git", "init", str(staging)])
            run(["git", "remote", "add", "origin", LAUNCHER_URL], cwd=staging)
            run(["git", "fetch", "--depth", "1", "origin", LAUNCHER_SHA], cwd=staging)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=staging)
            require(run(["git", "rev-parse", "HEAD"], cwd=staging, capture=True) == LAUNCHER_SHA,
                    "Fetched launcher commit does not match the pin.")
            require(not repo.exists(), "Another process created launcher; inspect before retrying.")
            staging.rename(repo)
        except BaseException:
            # Contains only our unexecuted download. Avoid following a replaced symlink.
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise
    recipe = work / "glm53-flash-2x128gb.yaml"
    if not recipe.exists():
        with urllib.request.urlopen(RECIPE_URL, timeout=60) as response:
            data = response.read(1024 * 1024)
        require(hashlib.sha256(data).hexdigest() == RECIPE_SHA256, "Downloaded recipe does not match checksum.")
        private_write(recipe, data.decode())
    verify_checkout(work)
    print("Pinned sources downloaded, not executed. Review launcher/setup-cluster.py, setup_cluster_node.py,\n"
          "launch-cluster.sh, run-recipe.py and docs/NETWORKING.md before running them.")

def parse_env(path):
    require(path.is_file() and not path.is_symlink(), "Missing cluster.env. Run configure-existing or cluster-apply first.")
    require(path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o077,
            "cluster.env must belong to you and have permissions 600.")
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        require("=" in line, "cluster.env has malformed syntax.")
        key, value = line.split("=", 1)
        require(key in ENV_KEYS and key not in values, "Unexpected/duplicate key in cluster.env; review it locally.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        require(not any(c in value for c in "$`\n\r\\;|&<>"), "Unsafe shell content in cluster.env.")
        values[key] = value
    return values

def validate_env(work, cfg):
    values = parse_env(work / "cluster.env")
    nodes = [ipv4(x) for x in values.get("CLUSTER_NODES", "").split(",")]
    expected = [ipv4(cfg["head_cluster_ip"]), ipv4(cfg["worker_cluster_ip"])]
    require(len(set(nodes)) == 2 and nodes == expected, "cluster.env must contain exactly your intended two cluster IPs, head first.")
    require(values.get("LOCAL_IP") == expected[0], "LOCAL_IP must match the head cluster IP.")
    require(re.fullmatch(r"[A-Za-z0-9_]+", values.get("ETH_IF", "")) is not None, "Invalid/missing ETH_IF.")
    require(re.fullmatch(r"=?[A-Za-z0-9_:]+(?:,[A-Za-z0-9_:]+)*", values.get("IB_IF", "")) is not None, "Invalid/missing IB_IF.")
    if "COPY_HOSTS" in values:
        hosts = [ipv4(x) for x in values["COPY_HOSTS"].split(",")]
        require(hosts == [cfg["worker_cluster_ip"]], "COPY_HOSTS must contain exactly the worker cluster IP for this direct-pair kit.")
    # This wrapper supports a direct pair on matching ports. Complex routing is left to upstream.
    require("CLUSTER_LINKS" not in values, "Complex/mixed-port routing is outside this kit. Use matching QSFP ports or upstream docs.")
    for key, value in values.items():
        if key.startswith("CONTAINER_"):
            require(value in {"none", "0", "1", "Ring"}, "Unexpected NCCL setting.")
    return values

def ssh_command(host, argv):
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
            host, shlex.join(list(map(str, argv)))]

def node_run(host, argv, capture=True):
    return run(ssh_command(host, argv), capture=capture, timeout=90)

def port_mapping(port):
    require(port in {"0", "1"}, "Explicit matching physical port 0 or 1 is required.")
    return (f"enp1s0f{port}np{port}", f"enP2p1s0f{port}np{port}"), (f"rocep1s0f{port}", f"roceP2p1s0f{port}")

NETWORK_PROBE = r"""import hashlib,json,pathlib,subprocess
identity=pathlib.Path('/etc/machine-id').read_text().strip()
if not identity: raise SystemExit('Missing machine identity')
links=json.loads(subprocess.check_output(['ip','-j','-4','addr','show'],text=True))
print(json.dumps({'node_id':hashlib.sha256(identity.encode()).hexdigest(),'links':links,'hcas':[p.name for p in pathlib.Path('/sys/class/infiniband').iterdir()]}))
"""

def validate_pair_network(cfg, head, worker, worker_via_cluster):
    """Validate only the operator's exact pair; derive no candidate hosts or subnets."""
    interfaces, hcas = port_mapping(cfg["ports"])
    expected = [ipv4(cfg["head_cluster_ip"]), ipv4(cfg["worker_cluster_ip"])]
    require(expected[0] != expected[1], "Head and worker cluster IPs must differ.")
    require(head.get("node_id") and worker.get("node_id") and head["node_id"] != worker["node_id"],
            "Management addresses must identify two distinct nodes.")
    require(worker_via_cluster.get("node_id") == worker["node_id"],
            "Worker cluster address does not identify the configured management node.")
    rails = []
    for info, first_ip, management_ip in zip((head, worker), expected,
            (cfg["head_management_ip"], cfg["worker_management_ip"])):
        links = {link["ifname"]: link for link in info["links"]}
        all_addresses = {entry.get("local") for link in info["links"] for entry in link.get("addr_info", [])}
        require(management_ip in all_addresses, "Configured management address is not present on its expected node.")
        require(set(hcas) <= set(info["hcas"]), "Expected RoCE interfaces for the selected physical port are missing.")
        node_rails = []
        for interface in interfaces:
            require(interface in links, "Selected physical port's Ethernet interface is missing.")
            link = links[interface]
            require(link.get("operstate") == "UP", "Both selected CX7 twin interfaces must be UP.")
            addresses = [entry for entry in link.get("addr_info", []) if entry.get("family") == "inet" and entry.get("scope") == "global"]
            require(len(addresses) == 1, "Each selected CX7 twin must have exactly one unambiguous global IPv4 address.")
            address = addresses[0]
            parsed = ipaddress.IPv4Interface(f"{ipv4(address['local'])}/{address['prefixlen']}")
            node_rails.append((parsed, link.get("mtu")))
        require(str(node_rails[0][0].ip) == first_ip,
                "Configured cluster IP is not on the first twin of the selected matching physical port.")
        rails.append(node_rails)
    for first, second in zip(*rails):
        require(first[0].network == second[0].network and first[0].ip != second[0].ip,
                "Each CX7 rail must use the same subnet across the pair and distinct node addresses.")
        require(type(first[1]) is int and first[1] == second[1] and first[1] >= 1500,
                "Peer MTUs differ or are invalid. Fix the existing cluster outside this kit.")
    require(rails[0][0][0].network != rails[0][1][0].network,
            "The two CX7 twins must use distinct rail subnets.")
    return {"CLUSTER_NODES": ",".join(expected), "COPY_HOSTS": expected[1], "LOCAL_IP": expected[0],
            "ETH_IF": interfaces[0], "IB_IF": ",".join(hcas)}

def configure_existing(work, cfg):
    destination = work / "cluster.env"
    require(not destination.exists() and not destination.is_symlink(),
            "cluster.env already exists; inspect it instead of overwriting a working configuration.")
    # Validate all operator choices before even making a bounded SSH probe.
    port_mapping(cfg["ports"])
    ipv4(cfg["head_cluster_ip"])
    ipv4(cfg["worker_cluster_ip"])
    print("Read-only checks of this head and ONLY the configured worker management/cluster addresses.\n"
          "No subnet scan, broadcast discovery, network changes or new SSH trust. Existing verified host keys are required.")
    probe = ["python3", "-c", NETWORK_PROBE]
    head = json.loads(run(probe, capture=True))
    worker = json.loads(node_run(cfg["worker_management_ip"], probe))
    worker_cluster = json.loads(node_run(cfg["worker_cluster_ip"], probe))
    values = validate_pair_network(cfg, head, worker, worker_cluster)
    # All checks must succeed before atomically creating the private final file.
    private_write(destination, "# Exact-pair settings; no discovery scan performed.\n" +
                  "".join(f"{key}={value}\n" for key, value in values.items()))
    print("Saved the validated exact pair to private cluster.env. No network/access settings changed.")

def check_docker_target(runner):
    probe = "import os,json; print(json.dumps([k for k in ('DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH') if os.environ.get(k)]))"
    overrides = json.loads(runner(["python3", "-c", probe]))
    require(not overrides, "Docker target overrides are present. Unset DOCKER_HOST/DOCKER_CONTEXT/TLS overrides in that node's login environment.")
    endpoint = runner(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"])
    require(endpoint == "unix:///var/run/docker.sock", "Active Docker context is not the local system socket. Select the intended local context on that Spark first.")

def check_head(cfg, require_cluster=True):
    require(os.geteuid() != 0, "Run as your normal user on the head Spark, not with sudo.")
    addresses = json.loads(run(["ip", "-j", "-4", "addr", "show"], capture=True))
    own = {v["local"] for x in addresses for v in x.get("addr_info", [])}
    require(cfg["head_management_ip"] in own, "Run this script on the configured head Spark, not your laptop/worker.")
    if require_cluster and cfg["head_cluster_ip"]:
        require(cfg["head_cluster_ip"] in own, "Head cluster IP is not local.")

def recipe_command(work, cfg, action):
    cmd = ["uv", "run", "--no-project", "--with", "pyyaml==6.0.2", "python", str(work / "launcher/run-recipe.py"),
           str(work / "glm53-flash-2x128gb.yaml"), "--config", str(work / "cluster.env"),
           "--name", cfg["container_name"], "--host", "127.0.0.1", "--port", str(cfg["api_port"]),
           "--master-port", str(cfg["master_port"]), "--no-ray", "--no-cache-dirs", "--max-model-len", str(cfg["max_model_len"])]
    if action == "setup":
        cmd.append("--setup")
    return cmd

def launch_command(work, cfg, action):
    return ["bash", str(work / "launcher/launch-cluster.sh"), "--config", str(work / "cluster.env"),
            "--name", cfg["container_name"], "--master-port", str(cfg["master_port"]), action]

def cluster_command(work, cfg, action):
    cmd = ["bash", str(work / "launcher/setup-cluster.sh"), "--state-file", str(work / "cluster-setup.json")]
    if action in {"cluster-doctor", "cluster-restore"}:
        cmd += ["--doctor", "--dry-run"] if action == "cluster-doctor" else ["--restore"]
        return cmd
    require(cfg["ports"] in {"0", "1"}, "Explicit matching physical port 0 or 1 is required before any cluster plan/apply.")
    try:
        pool = ipaddress.IPv4Network(cfg["subnet_pool"], strict=True)
    except (ValueError, TypeError) as exc:
        raise KitError("Set subnet_pool to a site-approved, unused RFC1918 range containing at least two /24s.") from exc
    private = [ipaddress.IPv4Network(x) for x in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
    require(pool.prefixlen <= 23 and any(pool.subnet_of(x) for x in private), "subnet_pool must be RFC1918 and /23 or larger.")
    require(all(ipaddress.IPv4Address(cfg[k]) not in pool for k in ("head_management_ip", "worker_management_ip")), "Subnet pool overlaps management addresses.")
    cmd += [cfg["head_management_ip"], cfg["worker_management_ip"], "--topology", "direct", "--mtu", "9000",
            "--subnet-pool", str(pool), "--env-file", str(work / "cluster.env")]
    cmd += ["--ports", cfg["ports"]]
    if action == "cluster-plan":
        cmd += ["--dry-run"]
    return cmd

PROBE = r'''import json, os, pathlib, platform, shutil, subprocess, sys
cache = pathlib.Path(sys.argv[1]); base = cache
while not base.exists(): base = base.parent
unsafe = []
for p in [cache, *cache.parents]:
    if p.is_symlink(): unsafe.append(str(p))
    if p in (cache, cache.parent) and p.exists() and (p.stat().st_uid != os.getuid() or p.stat().st_mode & 0o077): unsafe.append(str(p))
mem = int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))) * 1024
print(json.dumps(dict(unsafe_cache_paths=unsafe, machine=platform.machine(), home=str(pathlib.Path.home()), mem_gib=round(mem/2**30,1), free_gib=round(shutil.disk_usage(base).free/2**30,1), tools={x:bool(shutil.which(x)) for x in ('docker','nvidia-smi','rsync','python3','ip','ping')})))'''

def preflight(work, cfg, idle=False, need_download=True):
    validate_env(work, cfg)
    check_head(cfg)
    require(shutil.which("uv") and shutil.which("uvx"), "Install uv from its official documentation first.")
    env = child_environment(work)
    for host, local in [(cfg["head_cluster_ip"], True), (cfg["worker_cluster_ip"], False)]:
        runner = (lambda cmd: run(cmd, capture=True, env=env)) if local else (lambda cmd: node_run(host, cmd))
        info = json.loads(runner(["python3", "-c", PROBE, str(work / "hf-cache")]))
        print(json.dumps({"node": host, **info}, indent=2))
        require(not info["unsafe_cache_paths"], "Work/cache paths must be nonsymlinked, private (700), and owned by the login user on both nodes.")
        require(info["machine"] in {"aarch64", "arm64"} and info["mem_gib"] >= 115, "Each node must be an ARM64 128GB-class Spark.")
        require(info["home"] == str(Path.home()), "Both nodes must use the same username and home path.")
        require(all(info["tools"].values()), "Required node tool missing; install it before proceeding.")
        floor = cfg["minimum_free_gib"] if need_download else 20
        require(info["free_gib"] >= floor, f"Insufficient cache-volume free space: need {floor} GiB for this stage. This is a planning floor, not a guarantee.")
        gpu = runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        require("GB10" in gpu, "Expected an NVIDIA GB10 GPU on each node.")
        check_docker_target(runner)
        all_names = runner(["docker", "ps", "-a", "--format", "{{.Names}}"])
        if idle:
            require(cfg["container_name"] not in all_names.splitlines(), "Named container already exists on a node. Inspect/stop it explicitly first.")
            ids = runner(["docker", "ps", "-q"]).splitlines()
            if ids:
                details = json.loads(runner(["docker", "inspect", *ids]))
                for container in details:
                    hc = container.get("HostConfig", {})
                    gpu_request = hc.get("Runtime") == "nvidia" or bool(hc.get("DeviceRequests")) or any("nvidia" in str(x) for x in hc.get("Devices", []) or [])
                    require(not gpu_request, "A GPU container is already running. Stop your other model yourself; this kit will not stop Qwen/other services.")
            pids = runner(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            require(not pids.strip(), "GPU processes exist or process inventory is unsupported; resolve before launching.")
            port_probe = "import socket,sys; s=socket.socket(); s.bind(('0.0.0.0',int(sys.argv[1]))); s.close()"
            runner(["python3", "-c", port_probe, str(cfg["master_port"])])
            if local:
                runner(["python3", "-c", port_probe, str(cfg["api_port"])])
        print(f"PASS: {host}: basic hardware, SSH and Docker checks.")
    print("This is not an RDMA bandwidth test or a guarantee that model startup will fit memory.")

def confirm(args, warning, required_flag=None):
    print(warning)
    if required_flag:
        require(getattr(args, required_flag), f"Read the warning and repeat with --{required_flag.replace('_','-')}.")
    if not args.yes:
        require(sys.stdin.isatty(), "Interactive confirmation required. Use --yes only after reviewing all effects.")
        require(input("Type APPLY to continue: ") == "APPLY", "Cancelled; no requested action was taken.")

def preparation_record(cfg):
    return {"nodes": [cfg["head_cluster_ip"], cfg["worker_cluster_ip"]], "model": MODEL,
            "launcher": LAUNCHER_SHA, "recipe_sha256": RECIPE_SHA256}

def check_prepared(work, cfg):
    path = work / "prepared.json"
    require(path.is_file() and not path.is_symlink(), "No completed preparation receipt. Run setup to synchronize both nodes first.")
    require(json.loads(path.read_text()) == preparation_record(cfg), "Preparation receipt is for another pair/source revision. Use a separate workspace.")

def mark_prepared(work, cfg):
    path = work / "prepared.json"
    if path.exists():
        check_prepared(work, cfg)
    else:
        private_write(path, json.dumps(preparation_record(cfg), indent=2) + "\n")

def prepare_cache(work, cfg):
    script = """import os,pathlib,sys
os.umask(0o077)
p=pathlib.Path(sys.argv[1])
for parent in [p,*p.parents]:
    if parent.is_symlink(): raise SystemExit('Symlink in cache path')
p.mkdir(mode=0o700,parents=True,exist_ok=True)
for q in (p,p.parent):
    if q.stat().st_uid != os.getuid() or q.stat().st_mode & 0o077: raise SystemExit('Unsafe cache/work ownership or mode')
"""
    run(["python3", "-c", script, str(work / "hf-cache")])
    node_run(cfg["worker_cluster_ip"], ["python3", "-c", script, str(work / "hf-cache")])

def pull_image(work):
    receipt = work / "image-lock.json"
    if receipt.exists():
        verify_image(work)
        return
    run(["docker", "pull", IMAGE])
    inspect = json.loads(run(["docker", "image", "inspect", IMAGE], capture=True))[0]
    private_write(receipt, json.dumps({"tag": IMAGE, "id": inspect["Id"], "repo_digests": inspect.get("RepoDigests", [])}, indent=2) + "\n")
    print("Recorded the pulled image identity in private image-lock.json. The upstream tag itself is mutable.")

def verify_image(work, cfg=None):
    receipt = work / "image-lock.json"
    require(receipt.is_file(), "No image receipt; use setup first.")
    lock = json.loads(receipt.read_text())
    actual = run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], capture=True)
    require(actual == lock["id"], "Local image changed since setup. Review before using a new image.")
    if cfg:
        remote = node_run(cfg["worker_cluster_ip"], ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE])
        if remote != actual:
            # Upstream normalizes containerd/classic image-store identity differences.
            helper = work / "launcher/docker/image_identity.py"
            head_json = run(["docker", "image", "inspect", actual], capture=True)
            worker_json = node_run(cfg["worker_cluster_ip"], ["docker", "image", "inspect", remote])
            head_fp = run(["python3", str(helper)], capture=True, input_text=head_json)
            worker_fp = run(["python3", str(helper)], capture=True, input_text=worker_json)
            require(head_fp == worker_fp, "Worker runnable image content differs. Re-run setup after review.")

def check_health(cfg, chat=True):
    base = f"http://127.0.0.1:{cfg['api_port']}"
    # Do not use HTTP(S)_PROXY for loopback API requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/health", timeout=10) as response:
        require(response.status == 200, "Health endpoint did not return HTTP 200.")
    with opener.open(base + "/v1/models", timeout=20) as response:
        data = json.load(response)
    models = [x.get("id") for x in data.get("data", [])]
    require(MODEL in models, "Endpoint does not report the expected GLM model. Check port and service identity.")
    print("PASS: health and expected model ID.")
    if chat:
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Reply with the word READY."}], "max_tokens": 64, "temperature": 0}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with opener.open(req, timeout=180) as response:
            output = json.load(response)
        require(output.get("choices"), "No completion choices returned.")
        print("PASS: chat request returned choices. Inspect output; this is a smoke test, not a quality benchmark.")
        print(json.dumps(output, indent=2))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "configure-existing", "check-config", "preflight", "cluster-plan", "cluster-apply", "cluster-doctor", "cluster-restore", "plan", "setup", "start", "status", "stop", "health", "tunnel"])
    parser.add_argument("--config", default="settings.local.json")
    parser.add_argument("--work-dir", default=str(Path.home() / ".local/share/spark-pair-kit"))
    parser.add_argument("--yes", action="store_true", help="Skip wrapper typing prompt; never skips upstream plan confirmation.")
    parser.add_argument("--ack-network-changes", action="store_true")
    parser.add_argument("--ack-runtime-risk", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    work = Path(os.path.abspath(os.path.expanduser(args.work_dir)))
    ensure_work(work)
    if args.action == "fetch":
        fetch(work)
        return 0
    tools_bin = work / "tools/bin"
    if tools_bin.is_dir():
        os.environ["PATH"] = str(tools_bin) + os.pathsep + os.environ.get("PATH", "")
    cfg = load_settings(args.config)
    if args.action == "tunnel":
        print("Run on your laptop, replacing LOGIN with the Spark username. Verify the SSH fingerprint first:")
        print(shlex.join(["ssh", "-o", "ExitOnForwardFailure=yes", "-o", "StrictHostKeyChecking=yes", "-N", "-L", f"127.0.0.1:{cfg['api_port']}:127.0.0.1:{cfg['api_port']}", f"LOGIN@{cfg['head_management_ip']}"]))
        print(f"Client base URL: http://127.0.0.1:{cfg['api_port']}/v1 ; model: {MODEL}")
        return 0
    if args.action == "health":
        check_health(cfg)
        return 0
    repo = verify_checkout(work)
    env = child_environment(work)
    if args.action.startswith("cluster-"):
        check_head(cfg, require_cluster=False)
        command = cluster_command(work, cfg, args.action)
        if args.action in {"cluster-apply", "cluster-restore"}:
            confirm(args, "NETWORK/ACCESS CHANGE: this upstream helper changes Netplan and applies it; creates or removes\n"
                    "persistent SSH keys/trust and authorized_keys; changes Docker group membership (root-equivalent).\n"
                    "Networking may drop. Have console/management access, site approval and a maintenance window.\n"
                    "Do NOT run on an already-working NVIDIA Sync cluster. Restore can also disconnect you.\n"
                    "Backups/rollback can fail if a node becomes unreachable. Read upstream docs and preserve journals.", "ack_network_changes")
        else:
            print("Inspection only: upstream may request SSH/sudo authentication and create temporary validation files.\n"
                  "Verify host fingerprints independently. No automatic package installation or security-warning bypass.")
        run(command, cwd=repo, env=env)
        print("Review upstream output. For a new cluster, set head_cluster_ip/worker_cluster_ip from private cluster.env,\n"
              "then start a fresh login shell to pick up Docker group membership. Do not share cluster.env/journals.")
        return 0
    if args.action == "configure-existing":
        check_head(cfg)
        configure_existing(work, cfg)
        return 0
    validate_env(work, cfg)
    if args.action == "check-config":
        print("PASS: exactly two approved nodes, head first, with safe network fields. No connectivity test performed.")
    elif args.action == "plan":
        print("Wrapper-only preview. The following is not executed (upstream dry-run can discover/write settings):")
        print("Preparation: pull/verify image; explicitly copy it to worker; force incremental model download/sync.")
        print(shlex.join(recipe_command(work, cfg, "start")))
        print("API binding is loopback. Context starts at 32768 unless you explicitly change it. TP=2 from official recipe.")
    elif args.action == "preflight":
        preflight(work, cfg, idle=True)
    elif args.action in {"setup", "start"}:
        if args.action == "start":
            check_prepared(work, cfg)
        preflight(work, cfg, idle=True, need_download=args.action == "setup")
        confirm(args, "RUNTIME CHANGE: execute community code and a GPU container with upstream privileged/host-network\n"
                "settings on BOTH Sparks, mount the kit HF cache, and consume most memory. Setup downloads/copies\n"
                "large model/image files and may repair cache ownership. Review model/container licenses.\n"
                "No other container is stopped, and existing model caches are not intentionally changed.\n"
                "API binds loopback, but distributed ports still need a trusted isolated network. This is not a sandbox.", "ack_runtime_risk")
        env = strict_ssh_environment(work, env)
        prepare_cache(work, cfg)
        if args.action == "setup":
            pull_image(work)
            # Upstream --setup can skip an existing head model or stale worker tag.
            # Explicit copy + incremental HF sync make interrupted setup resumable.
            run(["bash", str(repo / "build-and-copy.sh"), "--no-build", "-t", IMAGE,
                 "--copy-to", cfg["worker_cluster_ip"], "--copy-parallel", "--config", str(work / "cluster.env")], cwd=repo, env=env)
            verify_image(work, cfg)
            download = recipe_command(work, cfg, "start") + ["--download-only", "--force-download"]
            run(download, cwd=repo, env=env)
            mark_prepared(work, cfg)
        else:
            verify_image(work, cfg)
        run(recipe_command(work, cfg, "start"), cwd=repo, env=env)
    elif args.action == "status":
        check_head(cfg)
        for host, local in [(cfg["head_cluster_ip"], True), (cfg["worker_cluster_ip"], False)]:
            command = ["docker", "ps", "-a", "--filter", "name=^/" + cfg["container_name"] + "$", "--format", "{{.Names}} {{.Status}} {{.Image}}"]
            runner = (lambda cmd: run(cmd, capture=True)) if local else (lambda cmd: node_run(host, cmd))
            check_docker_target(runner)
            print(host + ": " + runner(command))
        print("Container status does not establish API readiness; run health separately.")
    elif args.action == "stop":
        check_head(cfg)
        confirm(args, "Stop ONLY the configured GLM container name on both approved nodes. Other services are untouched.\n"
                "The upstream --rm container's writable layer is discarded; model/cache bind mounts remain.")
        # Do not trust upstream stop's success message: stop exactly this name, then independently verify.
        for host, local in [(cfg["head_cluster_ip"], True), (cfg["worker_cluster_ip"], False)]:
            runner = (lambda cmd: run(cmd, capture=True)) if local else (lambda cmd: node_run(host, cmd))
            check_docker_target(runner)
            names = runner(["docker", "ps", "--format", "{{.Names}}"])
            if cfg["container_name"] in names.splitlines():
                runner(["docker", "stop", "--time", "30", cfg["container_name"]])
            require(cfg["container_name"] not in runner(["docker", "ps", "--format", "{{.Names}}"]).splitlines(), "Stop verification failed.")
        print("Verified: configured GLM container is not running on either node.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KitError, OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted. Check status on both nodes; upstream cleanup is best-effort.", file=sys.stderr)
        sys.exit(130)
