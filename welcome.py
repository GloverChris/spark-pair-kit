#!/usr/bin/env python3
"""Interactive entrypoint. No silent network changes and no credential collection."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import spark_pair as kit

BANNER = r'''
             .                       .
            /|\                     /|\
       .---[ S P A R K ]===========[ S P A R K ]---.
       |          128 GB    +    128 GB           |
       '-------------------+---------------------'
                           |
                    .------v------.
                    |  GLM  5.3   |
                    |    FLASH    |
                    '-------------'
           TWO SPARKS. ONE GUIDED LAUNCH.
'''

def yes(prompt, default=False):
    suffix = " [Y/n] " if default else " [y/N] "
    reply = input(prompt + suffix).strip().lower()
    return default if not reply else reply in {"y", "yes"}

def ask(prompt, default=None):
    value = input(prompt + (f" [{default}]" if default is not None else "") + ": ").strip()
    return value or default

def ask_ip(prompt, default=None):
    while True:
        try:
            return kit.ipv4(ask(prompt, default))
        except kit.KitError as exc:
            print(exc)

def ask_port():
    while True:
        port = ask("Verified physical QSFP port number on BOTH Sparks (0 or 1)")
        if port in {"0", "1"}:
            return port
        print("Choose 0 or 1 only, after checking the cable. Automatic or mixed-port selection is not supported.")

def save_config(path, cfg):
    # Only this explicitly selected local settings file is updated by the wizard.
    kit.require(not path.is_symlink(), "Settings file must not be a symlink.")
    if path.exists():
        kit.require(path.stat().st_uid == os.getuid(), "Settings must belong to you.")
    fd, temporary = kit.tempfile.mkstemp(prefix=".settings-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(json.dumps(cfg, indent=2) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def bootstrap_tools(work):
    packages = []
    if not shutil.which("git"):
        packages.append("git")
    probe = subprocess.run([sys.executable, "-c", "import venv, ensurepip"], capture_output=True)
    if probe.returncode:
        packages.append("python3-venv")
    if packages:
        print("Missing head-node tools:", ", ".join(packages))
        kit.require(yes("Install these Ubuntu packages with sudo? This updates apt metadata first."), "Install prerequisites and rerun setup.sh.")
        kit.run(["sudo", "apt-get", "update"])
        kit.run(["sudo", "apt-get", "install", "-y", *packages])
    tools_bin = work / "tools/bin"
    if tools_bin.is_dir():
        os.environ["PATH"] = str(tools_bin) + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("uv") or not shutil.which("uvx"):
        print("uv will be installed from PyPI in a private virtual environment, not into system Python.")
        kit.require(yes("Install the uv helper package?"), "Install uv using its official guide and rerun setup.sh.")
        kit.run([sys.executable, "-m", "venv", str(work / "tools")])
        # Version pinned for repeatable bootstrap, not a claim that all transitive tools are pinned.
        kit.run([str(tools_bin / "python"), "-m", "pip", "install", "uv==0.9.7"])
        os.environ["PATH"] = str(tools_bin) + os.pathsep + os.environ.get("PATH", "")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Guided two-Spark GLM setup, with explicit approvals.")
    parser.add_argument("--work-dir", default=str(Path.home() / ".local/share/spark-pair-kit"))
    parser.add_argument("--config", default="settings.local.json")
    parser.add_argument("--banner-only", action="store_true", help="Preview welcome art without inspecting/changing the machine.")
    args = parser.parse_args(argv)
    print(BANNER)
    print("Independent helper following NVIDIA's playbooks. Not an NVIDIA product.\n")
    if args.banner_only:
        return 0
    kit.require(sys.stdin.isatty(), "Open a terminal on the HEAD Spark and run bash setup.sh.")
    kit.require(sys.version_info >= (3, 10), "Python 3.10+ is required.")
    kit.require(platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}, "Run on a DGX Spark, not your laptop.")
    kit.require(os.geteuid() != 0, "Run as your regular login user, without sudo.")
    os.umask(0o077)
    work = Path(os.path.abspath(os.path.expanduser(args.work_dir)))
    config = Path(args.config).absolute()
    kit.ensure_work(work)
    print("You'll need two 128GB DGX Sparks, a direct QSFP cable on matching ports,\n"
          "working management LAN access, the same login username on both, and internet.\n"
          "Driver, firmware and Docker installation remain DGX OS/admin prerequisites.\n"
          "Downloads are large; allow at least 300 GiB free on each cache volume.\n"
          "No existing model is stopped or deleted. Pause other GPU workloads yourself.\n")
    if config.exists():
        cfg = kit.load_settings(config)
        print("Resuming your private settings:", config)
        print("Head:", cfg["head_management_ip"], "Worker:", cfg["worker_management_ip"])
        kit.require(yes("Are these the intended two devices?", True), "Edit the private settings file or choose --config with another path.")
    else:
        cfg = kit.DEFAULTS.copy()
        print("Use MANAGEMENT LAN addresses here, not the QSFP/RDMA addresses.")
        cfg["head_management_ip"] = ask_ip("Head management IPv4 (this Spark)")
        cfg["worker_management_ip"] = ask_ip("Worker management IPv4")
        kit.require(cfg["head_management_ip"] != cfg["worker_management_ip"], "Use two distinct devices.")
        save_config(config, cfg)
    kit.check_head(cfg, require_cluster=False)
    print("\n[1/5] Helper tools and pinned upstream sources")
    bootstrap_tools(work)
    print("Sources will be downloaded from GitHub at fixed commits, then the selected upstream code will run.\n"
          "Review the README and downloaded sources. Only continue if your workplace permits this software.")
    kit.require(yes("Download/reuse the pinned sources and continue?", True), "Cancelled.")
    kit.fetch(work)
    base = ["--config", str(config), "--work-dir", str(work)]
    def command(action, *flags):
        return kit.main([action, *base, *flags])
    print("\n[2/5] Cluster connection")
    envfile = work / "cluster.env"
    if not envfile.exists():
        existing = yes("Is the pair already clustered successfully (for example with NVIDIA Sync)?", True)
        cfg["ports"] = ask_port()
        if existing:
            print("Keeping the existing network and SSH setup. No discovery scan will run.\n"
                  "Read the verified primary CX7 addresses from NVIDIA Sync or your existing cluster settings.\n"
                  "We will read only this head and the worker's exact management/cluster addresses.\n"
                  "Verify the worker SSH fingerprints independently first; unknown/changed keys are refused.")
            cfg["head_cluster_ip"] = ask_ip("Verified head primary CX7 IPv4", cfg["head_cluster_ip"])
            cfg["worker_cluster_ip"] = ask_ip("Verified worker primary CX7 IPv4", cfg["worker_cluster_ip"])
            save_config(config, cfg)
            command("configure-existing")
        else:
            print("New direct-pair cluster. Ask your network administrator for an unused private subnet pool\n"
                  "containing at least two /24 networks. Do not guess a range used by your company/VPN.")
            cfg["subnet_pool"] = ask("Approved subnet pool, in CIDR notation")
            save_config(config, cfg)
            print("First we will preview only. Upstream may ask for SSH/sudo credentials; enter them there, not in settings.")
            command("cluster-plan")
            print("\nThe next action changes network configuration, persistent SSH access and Docker membership.")
            kit.require(yes("Have you reviewed the plan, obtained approval, and arranged console/recovery access?"), "No cluster changes applied. Rerun when ready.")
            command("cluster-apply", "--ack-network-changes")
            print("If Docker group membership changed, log out/in on the head, then rerun setup.sh.\n"
                  "This wizard resumes from its saved configuration; it does not reapply networking.")
    values = kit.parse_env(envfile)
    nodes = values.get("CLUSTER_NODES", "").split(",")
    kit.require(len(nodes) == 2, "Configuration contains something other than two nodes. Review cluster.env before proceeding.")
    print("Configured cluster head:", nodes[0], "worker:", nodes[1])
    print("Interfaces:", values.get("ETH_IF"), "/", values.get("IB_IF"))
    kit.require(yes("Do those addresses/interfaces match this exact pair?"), "No model launch. Review the private cluster.env.")
    cfg["head_cluster_ip"], cfg["worker_cluster_ip"] = nodes
    save_config(config, cfg)
    command("check-config")
    resume = False
    if (work / "prepared.json").exists():
        kit.check_prepared(work, cfg)
        resume = yes("Image/model preparation completed before. Start without downloading/syncing again?", True)
    print("\n[3/5] Preflight")
    print("The next checks require known, verified SSH host keys for the worker's cluster IP.\n"
          "If verification fails, independently verify that Spark's host fingerprint, connect to it with\n"
          "ssh from this head, and rerun. Never bypass a changed-host-key warning.")
    kit.preflight(work, cfg, idle=True, need_download=not resume)
    print("\n[4/5] Preview")
    command("plan")
    print("\n[5/5] Download, synchronize, and serve")
    print("Keep this terminal open. Startup can take a long time. The API is not ready until health checks pass.\n"
          "In a second terminal, use ./serve-glm.sh health with the same --config/--work-dir,\n"
          "or use ./serve-glm.sh tunnel to print the laptop connection command.")
    command("start" if resume else "setup", "--ack-runtime-risk")
    print("Server command ended. Inspect status on both nodes before restarting.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (kit.KitError, OSError, ValueError, kit.urllib.error.URLError) as exc:
        print(f"\nStopped safely: {exc}\nFix the reported issue, then rerun setup.sh to resume.\n"
              "If a cluster apply/restore or launch had started, verify both nodes; rollback/cleanup is best-effort.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Check both nodes before resuming; cleanup is best-effort.", file=sys.stderr)
        sys.exit(130)
