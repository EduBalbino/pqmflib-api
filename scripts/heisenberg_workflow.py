"""Local performance evidence and TIMINGS.md → GitHub PR synchronization."""

import hashlib
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def run(*args):
    return subprocess.check_output(args, text=True).strip()


ROOT = Path(run("git", "rev-parse", "--show-toplevel"))
STATE = Path(run("git", "rev-parse", "--absolute-git-dir")) / "heisenberg-performance"
SOURCE_PATHS = (
    "pqfmlib/experimental",
    "pqfmlib/maps/heisenberg.py",
    "pqfmlib/core/cudaq_execution.py",
    "pqfmlib/core/execution.py",
    "pqfmlib/core/framework.py",
    "pqfmlib/core/observables.py",
    "experiments/sia/benchmark_heisenberg.py",
    "experiments/sia/heisenberg_reference.py",
    "experiments/sia/heisenberg_model.py",
)


def sources(revision=None):
    paths = (run("git", "ls-tree", "-r", "--name-only", revision, "--", *SOURCE_PATHS)
             if revision is not None else
             run("git", "ls-files", "--cached", "--others", "--exclude-standard",
                 "--", *SOURCE_PATHS)).splitlines()
    return {path: hashlib.sha256(
        subprocess.check_output(["git", "show", f"{revision}:{path}"])
        if revision is not None else (ROOT / path).read_bytes()
    ).hexdigest() for path in paths if path.endswith(".py")}


def read(name):
    path = STATE / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Missing {name} measurement; run just reference, then just benchmark")
    return json.loads(path.read_text())


def write(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(report):
    elapsed = report.get("tilelang_ms")
    errors = report.get("per_row_error", [])
    if (report.get("status") != "passed" or report.get("rows") != 35300
            or report.get("threshold") != 1e-9 or len(errors) != 35300
            or any(e is None or not math.isfinite(e) or not 0 <= e <= 1e-9 for e in errors)
            or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0):
        raise ValueError("Need a timed, passing exact comparison on all 35,300 rows")


def compare(previous, candidate):
    for report in (previous, candidate):
        validate(report)
    for key in ("identity", "scope", "gpu", "environment"):
        if previous[key] != candidate[key]:
            raise ValueError(f"Measurements differ in {key}; remeasure the committed reference")
    if candidate["tilelang_ms"] > previous["tilelang_ms"] * 0.90:
        raise ValueError("Improvement blocked: runtime must fall by at least 10%")


def promotion_commits(promotion):
    """Read the explicit commit transition in an accepted measurement."""
    return promotion["previous_commit"], promotion["candidate_commit"]


def is_ancestor(revision, descendant):
    return subprocess.run(["git", "merge-base", "--is-ancestor", revision, descendant]).returncode == 0


def check():
    previous = read("reference")
    current, committed = sources(), sources("HEAD")
    if current == committed == previous["sources"]:
        print("No uncommitted Heisenberg source candidate; performance gate skipped.")
        return None
    if previous["sources"] != committed:
        raise ValueError("Reference does not measure HEAD; run just reference before editing")
    candidate = read("candidate")
    if candidate["sources"] != current:
        raise ValueError("Working code differs from measured code; remeasure")
    compare(previous, candidate)
    print(f"Passed: {previous['tilelang_ms'] / candidate['tilelang_ms']:.3f}× previous TileLang")
    return candidate


def validate_timings(text, candidate):
    """Return the newest numbered section after checking it documents candidate."""
    sections = re.split(r"(?=^## #\d+\s)", text, flags=re.MULTILINE)
    if len(sections) < 2 or not sections[0].lstrip().startswith("|"):
        raise ValueError("TIMINGS.md must start with its measurement table")
    rows = []
    for line in text.splitlines():
        match = re.match(
            r"\|\s*#(\d+)\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)×\s*\|",
            line,
        )
        if match:
            rows.append((int(match[1]), match[2].strip(), float(match[3]), float(match[4])))
    if not rows:
        raise ValueError("TIMINGS.md has no numbered measurement rows")
    number, label, elapsed, speedup = rows[-1]
    expected_elapsed = candidate["tilelang_ms"] / 1000
    expected_speedup = candidate["cudaq_ms"] / candidate["tilelang_ms"]
    if ("TileLang" not in label or abs(elapsed - expected_elapsed) >= 0.0051
            or abs(speedup - expected_speedup) >= 0.0051):
        raise ValueError("Update the latest TIMINGS.md row to match the passing candidate")

    section = next((part for part in sections
                    if re.match(rf"^## #{number}\s", part)), None)
    required = ("Was this", "With this math", "Became this")
    if (section is None or not all(term in section for term in required)
            or not re.search(r"```math\s*.+?```", section, flags=re.DOTALL)):
        raise ValueError(
            f"Add ## #{number} with Was this → With this math → Became this and a fenced math equation"
        )
    return number, section


def validate_pr_timings(text, comments, user):
    """Require the passing candidate and its mathematical explanation in TIMINGS.md."""
    promotion_path = STATE / "promotion.json"
    if promotion_path.exists():
        promotion = json.loads(promotion_path.read_text())
        previous, candidate = promotion["previous"], promotion["candidate"]
        previous_commit, candidate_commit = promotion_commits(promotion)
        if (previous["sources"] != sources(previous_commit)
                or candidate["sources"] != sources(candidate_commit)
                or not is_ancestor(candidate_commit, "HEAD")):
            raise ValueError("Accepted PR measurement is not in this branch; remeasure")
        compare(previous, candidate)
    else:
        previous, candidate = read("reference"), read("candidate")
        already_accepted = (
            previous["sources"] == candidate["sources"] == sources("HEAD")
            and previous["tilelang_ms"] == candidate["tilelang_ms"]
        )
        headings = {part.splitlines()[0] for part in re.split(
            r"(?=^## #\d+\s)", text, flags=re.MULTILINE)[1:] if part.strip()
        }
        already_posted = any(
            comment["user"]["login"] == user
            and any(comment["body"].startswith(heading) for heading in headings)
            for page in comments for comment in page
        )
        if not (already_accepted and already_posted):
            raise ValueError("No accepted post-commit improvement; commit through the performance hook first")
    if candidate["sources"] != sources():
        raise ValueError("PR candidate is stale; run just benchmark on the current code")

    validate_timings(text, candidate)


def benchmark(reference=False):
    before = sources()
    if reference and before != sources("HEAD"):
        raise ValueError("Measure the committed runtime before editing it")
    sys.path.insert(0, str(ROOT))
    from experiments.sia.benchmark_heisenberg import run_benchmark

    environment = sorted((d.metadata["Name"], d.version) for d in distributions())
    report = run_benchmark()
    validate(report)
    if sources() != before:
        raise ValueError("Source changed during measurement; result discarded")
    report.update(sources=before, environment=environment)
    write("reference" if reference else "candidate", report)


def pr(repo, number):
    text = (ROOT / "TIMINGS.md").read_text()
    parts = re.split(r"(?=^## #\d+\s)", text, flags=re.MULTILINE)
    if len(parts) < 2 or not parts[0].lstrip().startswith("|"):
        raise ValueError("TIMINGS.md must start with a table followed by ## #N descriptions")
    endpoint = f"repos/{repo}/issues/{number}/comments"
    comments = json.loads(run("gh", "api", "--paginate", "--slurp", endpoint))
    user = run("gh", "api", "user", "--jq", ".login")
    validate_pr_timings(text, comments, user)
    with tempfile.TemporaryDirectory() as directory:
        body = Path(directory) / "body.md"
        body.write_text(parts[0].strip() + "\n")
        subprocess.run(["gh", "pr", "edit", number, "--repo", repo,
                        "--body-file", str(body)], check=True)
        for section in parts[1:]:
            heading = section.splitlines()[0]
            number_match = re.match(r"## #(\d+)\s", heading)[1]
            prefix = f"## #{number_match} "
            matches = [c for page in comments for c in page
                       if c["user"]["login"] == user and c["body"].startswith(prefix)]
            if len(matches) > 1:
                raise ValueError(f"Multiple comments for #{number_match}; resolve before syncing")
            body.write_text(json.dumps({"body": section.strip() + "\n"}))
            target = f"repos/{repo}/issues/comments/{matches[0]['id']}" if matches else endpoint
            subprocess.run(["gh", "api", "--method", "PATCH" if matches else "POST",
                            target, "--input", str(body)], check=True, stdout=subprocess.DEVNULL)


def publish_improvement(repo, number):
    """Validate, commit measured source, push its PR branch, and sync TIMINGS."""
    branch = run("git", "branch", "--show-current")
    tracking = run("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    tracking_remote, tracking_branch = tracking.split("/", 1)
    remote_head = run("git", "rev-parse", "@{u}")
    remote_url = run("git", "remote", "get-url", "--push", tracking_remote)
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    expected_url = f"https://github.com/{repo}"
    if tracking_branch != branch or remote_url.rstrip("/").lower() != expected_url.lower():
        raise ValueError("Configure this branch to track the PR head repository before just promote")
    pr_head = json.loads(run("gh", "pr", "view", str(number), "--repo", repo,
                             "--json", "headRefName,headRefOid"))
    if branch != pr_head["headRefName"]:
        raise ValueError(f"Current branch {branch!r} is not PR head {pr_head['headRefName']!r}")

    text = (ROOT / "TIMINGS.md").read_text()
    promotion_path = STATE / "promotion.json"
    promotion = json.loads(promotion_path.read_text()) if promotion_path.exists() else None
    if (promotion and promotion["candidate"]["sources"] == sources("HEAD")
            and promotion["candidate"]["sources"] == sources()):
        previous, candidate = promotion["previous"], promotion["candidate"]
        previous_commit, candidate_commit = promotion_commits(promotion)
        if (previous["sources"] != sources(previous_commit)
                or candidate["sources"] != sources(candidate_commit)
                or candidate["sources"] != sources()
                or not is_ancestor(candidate_commit, "HEAD")
                or not is_ancestor(remote_head, "HEAD")):
            raise ValueError("Accepted improvement does not match this branch or its upstream; inspect before retrying")
        compare(previous, candidate)
        validate_timings(text, candidate)
        if pr_head["headRefOid"] != remote_head:
            raise ValueError("PR head changed since this improvement was accepted")
    else:
        if remote_head != run("git", "rev-parse", "HEAD") or pr_head["headRefOid"] != remote_head:
            raise ValueError("Local branch and PR head must match before promoting a new candidate")
        candidate = check()
        previous = read("reference")
        number_in_timings, section = validate_timings(text, candidate)
        if previous["sources"] != sources("HEAD"):
            raise ValueError("Reference does not match current HEAD; run just reference before editing")
        changed = [path for path, digest in candidate["sources"].items()
                   if previous["sources"].get(path) != digest]
        if not changed:
            raise ValueError("Passing candidate has no changed tracked Python source to commit")
        staged = set(run("git", "diff", "--cached", "--name-only").splitlines())
        if staged and (
            staged != set(changed)
            or subprocess.run(["git", "diff", "--cached", "--quiet", "--", *changed]).returncode
        ):
            raise ValueError("Staged changes must exactly match the measured candidate to retry just promote")

        heading = section.splitlines()[0]
        title_match = re.match(r"## #\d+\s+[—-]\s+(.+)", heading)
        if not title_match:
            raise ValueError("TIMINGS section heading must be `## #N — description`")
        title = f"Heisenberg #{number_in_timings}: {title_match[1].strip()}"
        body = "\n".join(line for line in section.splitlines()[1:]
                          if line.strip()[:3] != "```").strip()

        if not staged:
            subprocess.run(["git", "add", "--", *changed], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", title, "-m", body], cwd=ROOT, check=True)
        promoted = json.loads(promotion_path.read_text())
        previous_commit, candidate_commit = promotion_commits(promoted)
        if (promoted["previous"]["sources"] != sources(previous_commit)
                or promoted["candidate"]["sources"] != sources(candidate_commit)
                or candidate_commit != run("git", "rev-parse", "HEAD")):
            raise ValueError("Post-commit acceptance hook did not promote the measured source")

    subprocess.run(["git", "push"], cwd=ROOT, check=True)
    pr(repo, str(number))


def main():
    command, *args = sys.argv[1:]
    if command == "install":
        hook = Path(run("git", "rev-parse", "--git-path", "hooks/pre-commit"))
        text = hook.read_text()
        original = 'call_lefthook run "pre-commit" "$@"'
        forced = 'call_lefthook run "pre-commit" --force "$@"'
        if original not in text and forced not in text:
            raise ValueError("Unrecognized Lefthook launcher; cannot enforce empty/amended commits")
        hook.write_text(text.replace(original, forced))
    elif command in ("reference", "benchmark"):
        benchmark(command == "reference")
    elif command == "check":
        check()
    elif command == "accept":
        if sources("HEAD^") == sources("HEAD"):
            print("No Heisenberg source change in this commit; acceptance skipped.")
            return
        previous, candidate = read("reference"), read("candidate")
        compare(previous, candidate)
        if previous["sources"] != sources("HEAD^") or candidate["sources"] != sources("HEAD"):
            raise ValueError("Committed sources do not match the measured improvement")
        write("promotion", {
            "previous": previous, "candidate": candidate,
            "previous_commit": run("git", "rev-parse", "HEAD^"),
            "candidate_commit": run("git", "rev-parse", "HEAD"),
        })
        write("reference", candidate)
    elif command == "pr":
        pr(*args)
    elif command == "publish":
        publish_improvement(*args)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
