"""Unit and orchestration tests: no network, sudo, Docker or GPU execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spark_pair as kit
import welcome

class KitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.config = self.root / "settings.local.json"
        self.cfg = kit.DEFAULTS | {"head_management_ip":"192.0.2.11", "worker_management_ip":"192.0.2.12", "head_cluster_ip":"198.51.100.11", "worker_cluster_ip":"198.51.100.12", "subnet_pool":"10.77.0.0/23", "ports":"0"}
        self.config.write_text(json.dumps(self.cfg))
        self.env = self.root / "cluster.env"
        self.env.write_text("CLUSTER_NODES=198.51.100.11,198.51.100.12\nLOCAL_IP=198.51.100.11\nETH_IF=enp1s0f0np0\nIB_IF=rocep1s0f0,roceP2p1s0f0\nCOPY_HOSTS=198.51.100.12\n")
        self.env.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_load_valid(self):
        self.assertEqual(kit.load_settings(self.config), self.cfg)

    def test_missing_addresses(self):
        self.config.write_text(json.dumps(kit.DEFAULTS))
        with self.assertRaises(kit.KitError): kit.load_settings(self.config)

    def test_container_injection_and_generic_names_rejected(self):
        for name in ("vllm_node", "spark_pair_a;id", "spark_pair_$(id)", "spark_pair_", "spark_pair_a.b"):
            self.config.write_text(json.dumps(self.cfg | {"container_name":name}))
            with self.assertRaises(kit.KitError): kit.load_settings(self.config)

    def test_bool_not_a_port(self):
        self.config.write_text(json.dumps(self.cfg | {"api_port":True}))
        with self.assertRaises(kit.KitError): kit.load_settings(self.config)

    def test_unknown_config_rejected(self):
        self.config.write_text(json.dumps(self.cfg | {"HF_TOKEN":"not-a-token"}))
        with self.assertRaises(kit.KitError): kit.load_settings(self.config)

    def test_env_requires_two_expected_nodes(self):
        self.env.write_text(self.env.read_text().replace("198.51.100.11,198.51.100.12", "198.51.100.12,198.51.100.11"))
        with self.assertRaises(kit.KitError): kit.validate_env(self.root, self.cfg)

    def test_env_valid(self):
        self.assertEqual(kit.validate_env(self.root, self.cfg)["LOCAL_IP"], self.cfg["head_cluster_ip"])

    def test_env_secrets_and_shell_rejected(self):
        original = self.env.read_text()
        for suffix in ("HF_TOKEN=private\n", "CONTAINER_HF_TOKEN=private\n", "IB_IF=$(id)\n", "ETH_IF=other\n"):
            self.env.write_text(original + suffix)
            with self.assertRaises(kit.KitError): kit.parse_env(self.env)

    def test_env_permissions_rejected(self):
        self.env.chmod(0o644)
        with self.assertRaises(kit.KitError): kit.parse_env(self.env)

    def test_unexpected_copy_host(self):
        self.env.write_text(self.env.read_text().replace("COPY_HOSTS=198.51.100.12", "COPY_HOSTS=203.0.113.20"))
        with self.assertRaises(kit.KitError): kit.validate_env(self.root, self.cfg)

    def test_no_complex_topology(self):
        self.env.write_text(self.env.read_text() + "CLUSTER_LINKS='[]'\n")
        with self.assertRaises(kit.KitError): kit.validate_env(self.root, self.cfg)

    def test_private_write_refuses_overwrite(self):
        with self.assertRaises(kit.KitError): kit.private_write(self.env, "bad")

    def test_work_directory_symlink_rejected(self):
        link = self.root / "link"; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(kit.KitError): kit.ensure_work(link)

    def test_recipe_command_loopback_isolated_and_no_daemon(self):
        cmd = kit.recipe_command(self.root, self.cfg, "start")
        self.assertEqual(cmd[cmd.index("--host")+1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--name")+1], "spark_pair_glm53")
        self.assertIn("--no-ray", cmd)
        self.assertIn("--no-project", cmd)
        self.assertIn("--no-cache-dirs", cmd)
        for item in ("--daemon", "--force-build", "--apply-mod", "--discover", "--setup"):
            self.assertNotIn(item, cmd)

    def test_cluster_plan_only(self):
        cmd = kit.cluster_command(self.root, self.cfg, "cluster-plan")
        self.assertIn("--dry-run", cmd)
        self.assertNotIn("--yes", cmd)
        self.assertEqual(cmd[cmd.index("--mtu")+1], "9000")
        self.assertEqual(cmd[cmd.index("--topology")+1], "direct")

    def test_cluster_pool_validation(self):
        for pool in (None, "203.0.113.0/23", "10.77.0.0/24", "10.77.0.1/23"):
            with self.assertRaises(kit.KitError): kit.cluster_command(self.root, self.cfg | {"subnet_pool":pool}, "cluster-plan")

    def test_restore_uses_journal_not_new_settings(self):
        cmd = kit.cluster_command(self.root, self.cfg, "cluster-restore")
        self.assertIn("--restore", cmd)
        for bad in ("--env-file", "--topology", "--subnet-pool", self.cfg["head_management_ip"]):
            self.assertNotIn(bad, cmd)

    def test_doctor_allows_missing_cluster_ip(self):
        info = json.dumps([{"addr_info":[{"local":self.cfg["head_management_ip"]}]}])
        with patch.object(kit.os, "geteuid", return_value=1000), patch.object(kit, "run", return_value=info):
            kit.check_head(self.cfg, require_cluster=False)
            with self.assertRaises(kit.KitError): kit.check_head(self.cfg)

    def test_copy_hosts_cannot_target_head(self):
        self.env.write_text(self.env.read_text().replace("COPY_HOSTS=198.51.100.12", "COPY_HOSTS=198.51.100.11"))
        with self.assertRaises(kit.KitError): kit.validate_env(self.root, self.cfg)

    def test_remote_docker_context_rejected(self):
        results = iter(["[]", "ssh://unexpected-host"])
        with self.assertRaises(kit.KitError): kit.check_docker_target(lambda cmd: next(results))

    def test_docker_host_override_rejected(self):
        with self.assertRaises(kit.KitError): kit.check_docker_target(lambda cmd: '["DOCKER_HOST"]')

    def test_local_docker_accepted(self):
        results = iter(["[]", "unix:///var/run/docker.sock"])
        kit.check_docker_target(lambda cmd: next(results))

    def test_failed_fetch_does_not_poison_destination(self):
        with patch.object(kit, "run", side_effect=kit.KitError("network interrupted")):
            with self.assertRaises(kit.KitError): kit.fetch(self.root)
        self.assertFalse((self.root / "launcher").exists())
        self.assertFalse(list(self.root.glob(".launcher-download-*")))

    def test_atomic_private_write(self):
        new = self.root / "saved"
        kit.private_write(new, "complete")
        self.assertEqual(new.read_text(), "complete")
        self.assertEqual(new.stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(self.root.glob(".partial-*")))

    def network_fixture(self, worker=False, port="0"):
        suffix = "12" if worker else "11"
        interfaces, hcas = kit.port_mapping(port)
        def link(name, address):
            return {"ifname": name, "operstate": "UP", "mtu": 9000,
                    "addr_info": [{"family":"inet", "scope":"global", "local":address, "prefixlen":24}]}
        return {"node_id":"worker-id" if worker else "head-id", "hcas":list(hcas),
                "links":[link("management", "192.0.2."+suffix),
                         link(interfaces[0], "198.51.100."+suffix),
                         link(interfaces[1], "203.0.113."+suffix)]}

    def test_existing_pair_valid_mapping(self):
        worker = self.network_fixture(True)
        values = kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)
        self.assertEqual(values["ETH_IF"], "enp1s0f0np0")
        self.assertEqual(values["COPY_HOSTS"], self.cfg["worker_cluster_ip"])

    def test_existing_pair_missing_worker_cluster_address(self):
        worker = self.network_fixture(True)
        worker["links"][1]["addr_info"] = []
        with self.assertRaises(kit.KitError): kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)

    def test_existing_pair_missing_worker_management_address(self):
        worker = self.network_fixture(True)
        worker["links"][0]["addr_info"] = []
        with self.assertRaises(kit.KitError): kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)

    def test_existing_pair_cross_ports_rejected(self):
        worker = self.network_fixture(True, "1")
        with self.assertRaises(kit.KitError): kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)

    def test_existing_pair_wrong_cluster_endpoint_rejected(self):
        worker = self.network_fixture(True)
        other = worker | {"node_id":"different-device"}
        with self.assertRaises(kit.KitError): kit.validate_pair_network(self.cfg, self.network_fixture(), worker, other)

    def test_existing_pair_mtu_mismatch_rejected(self):
        worker = self.network_fixture(True)
        worker["links"][2]["mtu"] = 1500
        with self.assertRaises(kit.KitError): kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)

    def test_configure_existing_contacts_only_exact_worker_addresses(self):
        self.env.unlink()
        worker = self.network_fixture(True)
        with patch.object(kit, "run", return_value=json.dumps(self.network_fixture())) as local, patch.object(kit, "node_run", return_value=json.dumps(worker)) as remote:
            kit.configure_existing(self.root, self.cfg)
        self.assertEqual(local.call_count, 1)
        self.assertEqual([call.args[0] for call in remote.call_args_list], [self.cfg["worker_management_ip"],self.cfg["worker_cluster_ip"]])
        kit.validate_env(self.root, self.cfg)
        self.assertEqual(self.env.stat().st_mode & 0o777, 0o600)

    def test_configure_failed_probe_leaves_no_env(self):
        self.env.unlink()
        with patch.object(kit, "run", return_value=json.dumps(self.network_fixture())), patch.object(kit, "node_run", side_effect=kit.KitError("SSH verification failed")):
            with self.assertRaises(kit.KitError): kit.configure_existing(self.root, self.cfg)
        self.assertFalse(self.env.exists())

    def test_configure_failed_validation_leaves_no_env(self):
        self.env.unlink()
        with patch.object(kit, "run", return_value=json.dumps(self.network_fixture())), patch.object(kit, "node_run", return_value=json.dumps(self.network_fixture(True,"1"))):
            with self.assertRaises(kit.KitError): kit.configure_existing(self.root, self.cfg)
        self.assertFalse(self.env.exists())

    def test_configure_existing_does_not_overwrite(self):
        original = self.env.read_text()
        with patch.object(kit, "run") as local, patch.object(kit, "node_run") as remote:
            with self.assertRaises(kit.KitError): kit.configure_existing(self.root, self.cfg)
        self.assertEqual(self.env.read_text(), original)
        local.assert_not_called(); remote.assert_not_called()

    def test_cluster_auto_and_mixed_ports_rejected_before_execution(self):
        for port in (None, "auto", "0,1", "1,0", "0,0"):
            with patch.object(kit, "run") as run:
                with self.assertRaises(kit.KitError): kit.cluster_command(self.root, self.cfg | {"ports":port}, "cluster-apply")
            run.assert_not_called()

    def test_existing_missing_port_refused_before_any_probe(self):
        self.env.unlink()
        with patch.object(kit, "run") as local, patch.object(kit, "node_run") as remote:
            with self.assertRaises(kit.KitError): kit.configure_existing(self.root, self.cfg | {"ports":None})
        local.assert_not_called(); remote.assert_not_called()

    def test_port_prompt_rejects_automatic_choice(self):
        with patch("builtins.input", side_effect=["auto", "0,1", "1"]):
            self.assertEqual(welcome.ask_port(), "1")

    def test_runtime_environment_removes_dangerous_overrides(self):
        with patch.dict(os.environ, {"VLLM_SPARK_EXTRA_DOCKER_ARGS":"--privileged", "DOTENV_CLUSTER_NODES":"bad", "CONTAINER_HF_TOKEN":"secret", "HF_HOME":"other", "BASH_ENV":"evil"}):
            env = kit.child_environment(self.root)
        self.assertEqual(env["HF_HOME"], str(self.root / "hf-cache"))
        for key in ("VLLM_SPARK_EXTRA_DOCKER_ARGS", "DOTENV_CLUSTER_NODES", "CONTAINER_HF_TOKEN", "BASH_ENV"):
            self.assertNotIn(key, env)

    def test_ssh_strict_override_precedes_upstream(self):
        env = kit.strict_ssh_environment(self.root, {"PATH":"/usr/bin:/bin"})
        self.assertTrue(env["PATH"].startswith(str(self.root / "strict-ssh")))
        self.assertIn(' -o StrictHostKeyChecking=yes "$@"', (self.root / "strict-ssh/ssh").read_text())
        self.assertEqual(kit.ssh_command("192.0.2.12", ["printf", "a b"])[-1], "printf 'a b'")

    def test_yes_does_not_bypass_network_ack(self):
        args = type("Args", (), {"yes":True, "ack_network_changes":False})()
        with self.assertRaises(kit.KitError): kit.confirm(args, "warning", "ack_network_changes")

    def test_plan_does_not_invoke_upstream(self):
        with patch.object(kit, "verify_checkout", return_value=self.root / "launcher"), patch.object(kit, "run") as run:
            kit.main(["plan", "--config", str(self.config), "--work-dir", str(self.root)])
            run.assert_not_called()

    def test_setup_explicit_sync_and_launch_sequence(self):
        calls = []
        with patch.object(kit, "verify_checkout", return_value=self.root / "launcher"), patch.object(kit, "preflight"), patch.object(kit, "confirm"), patch.object(kit, "pull_image"), patch.object(kit, "prepare_cache"), patch.object(kit, "verify_image"), patch.object(kit, "run", side_effect=lambda c, **kw: calls.append(c)):
            kit.main(["setup", "--config", str(self.config), "--work-dir", str(self.root), "--yes", "--ack-runtime-risk"])
        self.assertEqual(len(calls), 3)
        self.assertIn("--copy-to", calls[0])
        self.assertIn("--no-build", calls[0])
        self.assertIn("--download-only", calls[1])
        self.assertIn("--force-download", calls[1])
        self.assertNotIn("--setup", calls[2])
        self.assertNotIn("--download-only", calls[2])

    def test_image_store_ids_may_differ_when_content_matches(self):
        (self.root / "image-lock.json").write_text(json.dumps({"id":"sha256:head"}))
        outputs = iter(["sha256:head", "head-inspect", "same-fingerprint", "same-fingerprint"])
        with patch.object(kit, "run", side_effect=lambda *a, **kw: next(outputs)), patch.object(kit, "node_run", side_effect=["sha256:worker", "worker-inspect"]):
            kit.verify_image(self.root, self.cfg)

    def test_different_image_content_rejected(self):
        (self.root / "image-lock.json").write_text(json.dumps({"id":"sha256:head"}))
        outputs = iter(["sha256:head", "head-inspect", "one", "two"])
        with patch.object(kit, "run", side_effect=lambda *a, **kw: next(outputs)), patch.object(kit, "node_run", side_effect=["sha256:worker", "worker-inspect"]):
            with self.assertRaises(kit.KitError): kit.verify_image(self.root, self.cfg)

    def test_preparation_receipt_is_pair_specific(self):
        kit.mark_prepared(self.root, self.cfg)
        kit.check_prepared(self.root, self.cfg)
        changed = self.cfg | {"worker_cluster_ip":"198.51.100.99"}
        with self.assertRaises(kit.KitError): kit.check_prepared(self.root, changed)

    def test_start_without_preparation_refused(self):
        with self.assertRaises(kit.KitError): kit.check_prepared(self.root, self.cfg)

    def test_welcome_art_only_no_side_effects(self):
        with patch.object(kit, "run") as run, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(welcome.main(["--banner-only"]), 0)
        run.assert_not_called()
        self.assertIn("TWO SPARKS", out.getvalue())

    def test_wizard_refuses_nonterminal(self):
        with patch.object(sys.stdin, "isatty", return_value=False), patch.object(kit, "run") as run:
            with self.assertRaises(kit.KitError): welcome.main([])
        run.assert_not_called()

    def test_captured_failure_diagnostics_redact_tokens(self):
        exc = subprocess.CalledProcessError(255, ["ssh"], stderr="Host key verification failed. hf_dummysecret")
        with patch.object(subprocess, "run", side_effect=exc):
            with self.assertRaises(kit.KitError) as caught: kit.run(["ssh"], capture=True)
        self.assertIn("Host key verification failed", str(caught.exception))
        self.assertNotIn("hf_dummysecret", str(caught.exception))

    def test_wizard_existing_cluster_never_calls_network_setup_or_discovery(self):
        self.env.unlink()
        actions = []
        def stage(argv):
            actions.append(argv[0])
            if argv[0] == "configure-existing":
                worker = self.network_fixture(True)
                values = kit.validate_pair_network(self.cfg, self.network_fixture(), worker, worker)
                kit.private_write(self.env, "".join(f"{key}={value}\n" for key, value in values.items()))
            return 0
        with patch.object(sys.stdin, "isatty", return_value=True), patch.object(welcome.platform, "system", return_value="Linux"), patch.object(welcome.platform, "machine", return_value="aarch64"), patch.object(kit.os, "geteuid", return_value=1000), patch.object(kit, "check_head"), patch.object(welcome, "bootstrap_tools"), patch.object(kit, "fetch"), patch.object(kit, "preflight"), patch.object(kit, "main", side_effect=stage), patch("builtins.input", side_effect=["y","y","y","0","","","y"]):
            welcome.main(["--config",str(self.config),"--work-dir",str(self.root)])
        self.assertEqual(actions, ["configure-existing","check-config","plan","setup"])

    def test_wizard_can_correct_saved_cluster_ip_typo(self):
        self.env.unlink()
        self.config.write_text(json.dumps(self.cfg | {"head_cluster_ip":"198.51.100.99"}))
        def early_check(cfg, require_cluster=True):
            if require_cluster and cfg["head_cluster_ip"] != self.cfg["head_cluster_ip"]:
                raise kit.KitError("Wrong cluster IP is not local")
        def stage(argv):
            if argv[0] == "configure-existing":
                current = kit.load_settings(self.config)
                self.assertEqual(current["head_cluster_ip"], self.cfg["head_cluster_ip"])
                worker = self.network_fixture(True)
                values = kit.validate_pair_network(current, self.network_fixture(), worker, worker)
                kit.private_write(self.env, "".join(f"{key}={value}\n" for key, value in values.items()))
            return 0
        with patch.object(sys.stdin, "isatty", return_value=True), patch.object(welcome.platform, "system", return_value="Linux"), patch.object(welcome.platform, "machine", return_value="aarch64"), patch.object(kit.os, "geteuid", return_value=1000), patch.object(kit, "check_head", side_effect=early_check), patch.object(welcome, "bootstrap_tools"), patch.object(kit, "fetch"), patch.object(kit, "preflight"), patch.object(kit, "main", side_effect=stage), patch("builtins.input", side_effect=["y","y","y","0",self.cfg["head_cluster_ip"],"","y"]):
            welcome.main(["--config",str(self.config),"--work-dir",str(self.root)])
        self.assertTrue(self.env.is_file())

    def test_wrapper_help_and_banner(self):
        root = Path(__file__).resolve().parents[1]
        for cmd in (["bash", str(root/"spark-pair.sh"), "--help"], ["bash", str(root/"setup.sh"), "--banner-only"]):
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
