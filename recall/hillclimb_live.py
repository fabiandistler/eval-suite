#!/usr/bin/env python3
"""Run check_live's sessions for one flow and write them in the hillclimb shape.

check_live.py prints a table and forgets the sessions. This runner keeps them:
one `results.jsonl` row and one `traces/<id>_rep<k>.json` transcript per
(case, rep), failed attempts in `errors.jsonl`, under `<flow>/<variant>/`, so
the claude-api report builder can compare layouts case by case. The session,
the plugin layouts and the routing observation are check_live's own; only the
recording, the grade and the run guards live here.

Guards, because every session costs real tokens:
- the served model must match `--model` (subagents excluded: they may run on
  another model by design), otherwise the attempt goes to `errors.jsonl`;
- the agents checkout must be clean, and every row records its commit, so two
  layouts compared in one report were measured on the same skills;
- the runner, check_live.py, inputs.json and `_state.json`'s `harness_paths`
  are hashed; a run refuses until a human has approved that hash with
  `--approve-harness`.

Stdlib-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import check_live as live

HERE = Path(__file__).resolve().parent
ATTEMPTS = 3

EXIT_INCOMPLETE = 1
EXIT_USAGE = 2
EXIT_HARNESS = 3
EXIT_NO_CLI = 4
EXIT_CHECKOUT = 5

EPILOG = """\
exit codes: 0 every case recorded, 1 some cases failed (see errors.jsonl),
  2 usage, 3 harness not approved, 4 claude CLI missing, 5 agents checkout dirty

examples:
  AGENTS_REPO=../agents/.worktrees/recall python3 recall/hillclimb_live.py \\
    --flow .claude/hillclimb/ai-ml-recall --variant baseline --layout routed \\
    --model claude-opus-5-5 --reps 5
  python3 recall/hillclimb_live.py --flow .claude/hillclimb/ai-ml-recall --approve-harness
"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def harness_sha(flow: Path, state: dict) -> str:
    """sha256 over the files that decide what a run measures and how it is scored."""
    files = [Path(__file__).resolve(), HERE / "check_live.py", flow / "inputs.json"]
    files += [flow / p for p in state.get("harness_paths", [])]
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def events(lines: list[str]) -> list[dict]:
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


def served_models(stream: list[dict]) -> set[str]:
    return {
        e["message"]["model"]
        for e in stream
        if e.get("type") == "assistant" and e.get("parent_tool_use_id") is None
    }


def transcript(stream: list[dict], prompt: str) -> list[dict]:
    """The session as report turns; stream-json does not echo the prompt, so it is inserted."""
    turns: list[dict] = []
    for e in stream:
        kind = e.get("type")
        if kind == "system" and e.get("subtype") == "init":
            listing = {k: e.get(k) for k in ("model", "plugins", "skills", "tools")}
            turns.append({"role": "system", "content": json.dumps(listing, indent=2)})
            turns.append({"role": "user", "content": prompt})
        elif kind == "assistant":
            for block in e["message"]["content"]:
                if block["type"] == "text":
                    turns.append({"role": "assistant", "content": block["text"]})
                elif block["type"] == "tool_use":
                    turns.append({
                        "role": "tool_call",
                        "name": block["name"],
                        "content": json.dumps(block["input"], indent=2, ensure_ascii=False),
                    })
                elif block["type"] == "thinking":
                    turns.append({"role": "assistant", "content": "", "thinking": block["thinking"]})
        elif kind == "user" and isinstance(e["message"]["content"], list):
            for block in e["message"]["content"]:
                if block.get("type") != "tool_result":
                    continue
                body = block.get("content")
                if isinstance(body, list):
                    body = "\n".join(str(b.get("text", "")) for b in body)
                turns.append({"role": "tool_result", "content": str(body)})
    return turns


def grade(expected: str, member: str | None, router_fired: bool) -> dict[str, float]:
    """`correct` is the headline; `fired` is any ai-ml entry opened, router or member."""
    fired = router_fired or member is not None
    if expected == "none":
        correct = not fired
    elif expected == "any":
        correct = fired
    else:
        correct = member == expected
    return {"correct": float(correct), "fired": float(fired)}


def run_case(case: dict, rep: int, ctx: dict) -> tuple[dict | None, list[dict]]:
    """One (case, rep): a recorded row, or None plus the failed attempts."""
    failures = []
    for attempt in range(1, ATTEMPTS + 1):
        started = time.monotonic()
        try:
            lines = live.run_prompt(case["prompt"], ctx["plugin_dirs"], ctx["workdir"], ctx["model"])
        except subprocess.TimeoutExpired:
            failure = "timeout"
        else:
            latency = time.monotonic() - started
            stream = events(lines)
            models = served_models(stream)
            if live.unusable(lines):
                failure = "unusable"
            elif not models or any(not m.startswith(ctx["model"]) for m in models):
                # A substituted model invalidates the layout comparison; never retried.
                failures.append(failure_row(case, rep, attempt, "model_mismatch", sorted(models)))
                return None, failures
            else:
                router_fired, member, via = live.observe(
                    lines, ctx["router"], ctx["members"], ctx["agents"]
                )
                result = next(e for e in stream if e.get("type") == "result")
                trace = ctx["vdir"] / "traces" / f"{case['id']}_rep{rep}.json"
                trace.write_text(
                    json.dumps(transcript(stream, case["prompt"]), indent=1, ensure_ascii=False),
                    encoding="utf-8",
                )
                return {
                    "prompt_id": case["id"],
                    "rep": rep,
                    "prompt": case["prompt"],
                    "tags": case["tags"],
                    "expected": case["expected"],
                    "status": "ok",
                    "stop_reason": result["subtype"],
                    "grade": grade(case["expected"], member, router_fired),
                    "model": sorted(models)[0],
                    "usage": result["usage"],
                    "cost_usd": result["total_cost_usd"],
                    "latency_s": round(latency, 1),
                    "turns": result["num_turns"],
                    "tool_calls": sum(
                        1
                        for e in stream
                        if e.get("type") == "assistant" and e.get("parent_tool_use_id") is None
                        for b in e["message"]["content"]
                        if b["type"] == "tool_use"
                    ),
                    "hit_max_turns": float(result["subtype"] == "error_max_turns"),
                    "meta": {
                        "layout": ctx["layout"],
                        "agents_commit": ctx["commit"],
                        "member": member,
                        "router_fired": router_fired,
                        "via": via,
                        "attempt": attempt,
                    },
                }, failures
        failures.append(failure_row(case, rep, attempt, failure, []))
        if attempt < ATTEMPTS:
            time.sleep(5 * 2**attempt + random.uniform(0, 5))
    return None, failures


def failure_row(case: dict, rep: int, attempt: int, cls: str, models: list[str]) -> dict:
    return {
        "prompt_id": case["id"],
        "rep": rep,
        "attempt": attempt,
        "class": cls,
        "served_models": models,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record check_live sessions as results.jsonl + traces/ for one flow variant.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--flow", required=True, type=Path, help="flow dir holding inputs.json and _state.json")
    parser.add_argument("--variant", help="baseline or v<N>")
    parser.add_argument("--layout", choices=("routed", "flat"), help="plugin layout for the category")
    parser.add_argument("--model", help="model id every session must be served by")
    parser.add_argument("--reps", type=int, default=1, help="runs per case (default 1)")
    parser.add_argument("--workers", type=int, default=4, help="parallel claude sessions (default 4)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--approve-harness", action="store_true", help="record the harness sha and exit (human only)")
    args = parser.parse_args()

    state_path = args.flow / "_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    sha = harness_sha(args.flow, state)
    if args.approve_harness:
        state["harness_sha"] = sha
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        sys.stderr.write(f"harness approved: sha256 {sha[:12]}\n")
        return 0
    if state.get("harness_sha") != sha:
        sys.stderr.write(
            f"harness sha {sha[:12]} is not the approved one; review the runner, check_live.py and "
            "inputs.json, then have a human run --approve-harness\n"
        )
        return EXIT_HARNESS
    if not (args.variant and args.layout and args.model):
        sys.stderr.write("--variant, --layout and --model are required for a run\n")
        return EXIT_USAGE
    if args.variant != "baseline" and not (args.variant[:1] == "v" and args.variant[1:].isdigit()):
        sys.stderr.write("--variant must be baseline or v<N>; the report ignores other names\n")
        return EXIT_USAGE
    if not live.MANIFEST_PATH.is_file():
        sys.stderr.write("no skills.json in the agents checkout; set AGENTS_REPO\n")
        return EXIT_USAGE
    if shutil.which("claude") is None:
        sys.stderr.write("claude CLI not on PATH\n")
        return EXIT_NO_CLI
    if git(live.REPO_ROOT, "status", "--porcelain", "--", "skills", "plugins", "skills.json"):
        sys.stderr.write("agents checkout has uncommitted changes under skills/ or plugins/\n")
        return EXIT_CHECKOUT
    commit = git(live.REPO_ROOT, "rev-parse", "HEAD")

    inputs = json.loads((args.flow / "inputs.json").read_text(encoding="utf-8"))
    category = inputs["category"]
    cases = inputs["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    vdir = args.flow / args.variant
    (vdir / "traces").mkdir(parents=True, exist_ok=True)
    results_path = vdir / "results.jsonl"
    recorded = read_jsonl(results_path)
    for row in recorded:
        if row["meta"]["layout"] != args.layout or row["meta"]["agents_commit"] != commit:
            sys.stderr.write(
                f"{results_path} holds rows from another layout or agents commit; use a new variant\n"
            )
            return EXIT_USAGE
    done = {(r["prompt_id"], r["rep"]) for r in recorded}
    jobs = [(c, r) for c in cases for r in range(args.reps) if (c["id"], r) not in done]

    skills = live.load_manifest()
    members = live.members_of(skills, category)
    router = live.router_of(skills, category)
    with tempfile.TemporaryDirectory(prefix="hillclimb-live-") as tmp_s:
        tmp = Path(tmp_s)
        plugin_dirs = [p for p in sorted(live.PLUGINS_DIR.iterdir()) if (p / ".claude-plugin").is_dir()]
        agents = live.agents_of(category, members)
        if args.layout == "flat":
            plugin_dirs = [p for p in plugin_dirs if p.name != category]
            plugin_dirs.append(live.make_flat_plugin(category, members, tmp))
            router = None
            agents = {}
        ctx = {
            "plugin_dirs": plugin_dirs,
            "workdir": live.make_workdir(tmp),
            "model": args.model,
            "router": router,
            "members": members,
            "agents": agents,
            "layout": args.layout,
            "commit": commit,
            "vdir": vdir,
        }
        sys.stderr.write(
            f"{category} {args.layout} -> {vdir} | {args.model} | agents {commit[:8]} | "
            f"{len(jobs)} runs to go, {len(done)} already recorded\n"
        )
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, c, r, ctx) for c, r in jobs]
            for fut in concurrent.futures.as_completed(futures):
                row, failures = fut.result()
                for f in failures:
                    append_jsonl(vdir / "errors.jsonl", f)
                if row is None:
                    failed += 1
                else:
                    append_jsonl(results_path, row)

    print(json.dumps({"variant": args.variant, "recorded": len(jobs) - failed, "failed": failed}))
    return EXIT_INCOMPLETE if failed else 0


if __name__ == "__main__":
    sys.exit(main())
