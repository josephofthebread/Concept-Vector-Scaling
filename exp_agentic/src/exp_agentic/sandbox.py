#! /usr/bin/env python


import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("sandbox")

BACKEND = os.environ.get("AGENTIC_SANDBOX", "docker")

IMAGE = "neuralprobes-agentic:1"
CPUS = "2"
MEMORY = "2g"
PIDS = 256
TIMEOUT = 30


@dataclass
class Result:
    """What one tool call produced."""

    code: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def output(self) -> str:
        """Both streams as the agent will see them, stderr last."""
        parts = [self.stdout.rstrip(), self.stderr.rstrip()]
        return "\n".join(part for part in parts if part)


class Sandbox:
    """A per-episode container over one writable directory."""

    def __init__(
        self,
        workdir: Path,
        image: str = IMAGE,
        cpus: str = CPUS,
        memory: str = MEMORY,
        pids: int = PIDS,
        timeout: int = TIMEOUT,
    ) -> None:
        self.workdir = workdir
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.timeout = timeout
        self.container: str | None = None

    def start(self) -> None:
        """Launch the container and leave it idling until tool calls arrive."""
        if self.container:
            raise RuntimeError("sandbox already started")
        self.workdir.mkdir(parents=True, exist_ok=True)
        if BACKEND == "subprocess":
            log.info(f"subprocess backend over {self.workdir}")
            return
        argv = [
            "docker",
            "run",
            "--detach",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=256m,exec",
            "--volume",
            f"{self.workdir.resolve()}:/work:rw",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids),
            "--workdir",
            "/work",
            self.image,
            "sleep",
            "infinity",
        ]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"docker run failed: {done.stderr.strip()}")
        self.container = done.stdout.strip()
        log.info(f"container {self.container[:12]} over {self.workdir}")

    def run(self, argv: list[str], timeout: int | None = None, cwd: str | None = None) -> Result:
        """Execute one command inside the container, without a shell."""
        limit = self.timeout if timeout is None else timeout

        if BACKEND == "subprocess":
            where = self.workdir.resolve()
            if cwd:
                where = where / cwd.removeprefix("/work").lstrip("/")
            try:
                done = subprocess.run(argv, capture_output=True, text=True, timeout=limit, cwd=str(where))
            except subprocess.TimeoutExpired:
                return Result(code=124, stdout="", stderr=f"timed out after {limit}s", timed_out=True)
            except FileNotFoundError as problem:
                return Result(code=127, stdout="", stderr=str(problem), timed_out=False)
            return Result(code=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)

        if not self.container:
            raise RuntimeError("sandbox not started")
        place = ["--workdir", cwd] if cwd else []
        wrapped = ["timeout", "--kill-after=2", str(limit), *argv]
        try:
            done = subprocess.run(
                ["docker", "exec", *place, self.container, *wrapped],
                capture_output=True,
                text=True,
                timeout=limit + 10,
            )
        except subprocess.TimeoutExpired:
            return Result(code=124, stdout="", stderr=f"docker exec wedged after {limit + 10}s", timed_out=True)
        if done.returncode in (124, 137):
            return Result(code=done.returncode, stdout=done.stdout, stderr=f"timed out after {limit}s", timed_out=True)
        return Result(code=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)

    def stop(self) -> None:
        """Remove the container. The bind-mounted directory is left in place as evidence."""
        if BACKEND == "subprocess" or not self.container:
            return
        subprocess.run(["docker", "rm", "--force", self.container], capture_output=True, text=True)
        log.info(f"container {self.container[:12]} removed; {self.workdir} kept")
        self.container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("episodes/sandbox-check"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        log.info(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures.append(name)

    with Sandbox(args.workdir) as box:
        version = box.run(["python", "--version"])
        check("python present", version.code == 0, version.output)

        pytest_version = box.run(["python", "-m", "pytest", "--version"])
        check("pytest present", pytest_version.code == 0, pytest_version.output)

        written = box.run(
            [
                "python",
                "-c",
                "open('/work/test_demo.py','w').write('def test_pass():\\n    assert 1 == 1\\n"
                "def test_fail():\\n    assert 2 == 3\\n')",
            ]
        )
        check("write into /work", written.code == 0, written.output or "(no output)")

        tests = box.run(["python", "-m", "pytest", "-q", "test_demo.py"])
        check(
            "pytest runs and reports a failure",
            tests.code != 0 and "1 failed" in tests.output and "1 passed" in tests.output,
            tests.output.strip().splitlines()[-1] if tests.output else "(no output)",
        )

        owned = (args.workdir / "test_demo.py").stat()
        check(
            "files owned by the host account",
            owned.st_uid == os.getuid(),
            f"uid {owned.st_uid} (host {os.getuid()})",
        )

        readonly = box.run(["python", "-c", "open('/usr/local/blocked','w')"])
        check(
            "root filesystem read-only",
            readonly.code != 0,
            readonly.stderr.strip().splitlines()[-1] if readonly.stderr else "(no error)",
        )

        tmp = box.run(["python", "-c", "open('/tmp/allowed','w').write('x')"])
        check("/tmp writable", tmp.code == 0, tmp.output or "(no output)")

        network = box.run(
            ["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"],
            timeout=15,
        )
        check(
            "network disabled",
            network.code != 0,
            network.stderr.strip().splitlines()[-1] if network.stderr else "(no error)",
        )

        limits = box.run(
            ["python", "-c", "import os; print(os.cpu_count(), open('/sys/fs/cgroup/memory.max').read().strip())"]
        )
        check("cgroup limits visible", limits.code == 0, limits.output)

        slept = box.run(["python", "-c", "import time; time.sleep(60)"], timeout=3)
        check("timeout fires", slept.timed_out, slept.stderr)

        survivors = box.run(["ps", "-eo", "args", "--no-headers"])
        check(
            "timed-out process actually died",
            "time.sleep(60)" not in survivors.stdout,
            "no survivors" if "time.sleep(60)" not in survivors.stdout else survivors.stdout.strip(),
        )

    if failures:
        raise SystemExit(f"sandbox checks failed: {', '.join(failures)}")
    log.info("every sandbox check passed")


if __name__ == "__main__":
    main()
