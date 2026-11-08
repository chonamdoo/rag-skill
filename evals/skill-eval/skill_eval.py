#!/usr/bin/env python3
"""Behavioral regression eval for a skill, run when the skill changes.

Runs every case against two arms with the same model and judge:
  head  the skill in the working tree
  base  the skill at --base-ref (skipped when the skill does not exist there)

Executors get the skill files in the system prompt and no tools. A separate
judge call, blind to the arm, marks each required criterion pass/fail. The run
fails when the head pass rate is below `min_pass_rate`, or more than `max_drop`
below the base arm. Results go to --out (JSON) and stdout.

    python3 evals/skill-eval/skill_eval.py evals/skill-eval/suite.json --base-ref origin/main

Requires a logged-in `claude` CLI.
Standard library only.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"criteria": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["index", "pass", "reason"]}}},
    "required": ["criteria"],
}
JUDGE_SYSTEM = (
    "You are a strict evaluator. You receive a user request, an assistant response and numbered acceptance "
    "criteria. Judge each criterion only from the response text. Pass a criterion only when the response "
    "clearly satisfies it; if a criterion does not apply to what the response did, explain why and pass it "
    "only when the criterion itself says that case is acceptable. Give a one-sentence reason per criterion."
)


class BudgetExceeded(RuntimeError):
    pass


class Runner:
    def __init__(self, max_usd, dry_run=False):
        self.max_usd, self.spent, self.dry_run = max_usd, 0.0, dry_run
        self.lock = threading.Lock()

    def claude(self, prompt, system, model, schema=None):
        with self.lock:
            if self.spent >= self.max_usd:
                raise BudgetExceeded(f"spent ${self.spent:.2f} >= max_usd ${self.max_usd:.2f}")
        if self.dry_run:
            return {"text": "", "structured": {"criteria": []}, "cost": 0.0}
        cmd = ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "",
               "--no-session-persistence", "--output-format", "json", "--system-prompt", system]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        last = None
        for _ in range(3):
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last = f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"
                continue
            with self.lock:
                self.spent += float(data.get("total_cost_usd") or 0)
            if data.get("is_error"):
                last = str(data.get("result"))[-500:]
                continue
            return {"text": data.get("result") or "", "structured": data.get("structured_output"),
                    "cost": float(data.get("total_cost_usd") or 0)}
        raise RuntimeError(f"claude call failed 3 times: {last}")


def skill_files_at_ref(skill_dir, ref):
    """{relative path: text} of the skill at a git ref, or None when absent there."""
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", skill_dir],
                             capture_output=True, text=True)
    if listing.returncode != 0 or not listing.stdout.strip():
        return None
    files = {}
    for path in listing.stdout.split():
        blob = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True)
        try:
            files[os.path.relpath(path, skill_dir)] = blob.stdout.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return files


def skill_files_worktree(skill_dir):
    files = {}
    for path in sorted(Path(skill_dir).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            try:
                files[str(path.relative_to(skill_dir))] = path.read_text("utf-8")
            except UnicodeDecodeError:
                continue
    return files


def skill_system_prompt(files):
    order = sorted(files, key=lambda p: (p != "SKILL.md", p))
    body = "\n\n".join(f'<file path="{p}">\n{files[p]}\n</file>' for p in order)
    return ("The following skill package is installed and applies to this conversation. Follow SKILL.md; "
            "read the referenced files below as needed. You have no tools: answer in text only.\n\n" + body)


def _criterion(item):
    if isinstance(item, str):
        return {"text": item, "required": True}
    return {"text": item["criterion"], "required": bool(item.get("required", True))}


def load_cases(suite, root):
    src = suite.get("cases_from")
    cases = [dict(c) for c in suite.get("cases", [])]
    if src:
        data = json.loads((root / src["path"]).read_text("utf-8"))
        fmt = src["format"]
        if fmt == "request-acceptance":
            cases += [{"id": c["id"], "request": c["request"], "criteria": [_criterion(a) for a in c["acceptance"]]}
                      for c in data]
        elif fmt == "input-expected":
            cases += [{"id": c["id"], "request": c["input"], "criteria": [_criterion(c["expected"])]}
                      for c in data["cases"]]
        elif fmt == "request-criteria-map":
            criteria = json.loads((root / src["criteria_path"]).read_text("utf-8"))
            for key in src.get("criteria_keys", []):
                criteria = criteria[key]
            cases += [{"id": c["id"], "request": c["request"], "criteria": [_criterion(t) for t in criteria[c["id"]]]}
                      for c in data if c["id"] in criteria]
        else:
            raise ValueError(f"unknown cases_from.format: {fmt}")
    for case in cases:
        case["criteria"] = [_criterion(c) if not isinstance(c, dict) or "text" not in c else c
                            for c in case["criteria"]]
    if suite.get("select"):
        wanted = set(suite["select"])
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise ValueError(f"select names unknown cases: {sorted(missing)}")
        cases = [c for c in cases if c["id"] in wanted]
    ids = [c["id"] for c in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("suite needs at least one case and unique case ids")
    return cases


def load_checks(spec, root):
    """Optional deterministic checks: module function (case_id, output) -> {criterion index: bool}."""
    if not spec:
        return None
    mod_spec = importlib.util.spec_from_file_location("suite_checks", root / spec["path"])
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return getattr(module, spec.get("function", "run"))


def judge_prompt(case, output):
    lines = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(case["criteria"]))
    return (f"<request>\n{case['request']}\n</request>\n\n<response>\n{output}\n</response>\n\n"
            f"<criteria>\n{lines}\n</criteria>\n\nReturn one entry per criterion index.")


def run_one(runner, suite, checks, arm, system, case, rep):
    out = runner.claude(case["request"], system, suite["model"])
    judged = runner.claude(judge_prompt(case, out["text"]), JUDGE_SYSTEM, suite["judge_model"], JUDGE_SCHEMA)
    marks = {e["index"]: e for e in (judged["structured"] or {}).get("criteria", [])}
    hard = {}
    if checks and not runner.dry_run:
        hard = {int(k): bool(v) for k, v in checks(case["id"], out["text"]).items()}
    results = []
    for i, crit in enumerate(case["criteria"]):
        mark = marks.get(i, {"pass": False, "reason": "judge returned no entry"})
        ok = bool(mark["pass"]) and hard.get(i, True)
        reason = mark["reason"] if hard.get(i, True) else f"string check failed; judge: {mark['reason']}"
        results.append({"index": i, "required": crit["required"], "pass": ok, "reason": reason})
    return {"arm": arm, "case": case["id"], "rep": rep, "output": out["text"],
            "cost": out["cost"] + judged["cost"], "criteria": results}


def rate(rows):
    marks = [c["pass"] for r in rows for c in r["criteria"] if c["required"]]
    return (sum(marks) / len(marks)) if marks else None


def summarize(suite, rows, arms, spent, verdict, reasons):
    head = [r for r in rows if r["arm"] == "head"]
    base = [r for r in rows if r["arm"] == "base"]
    lines = [f"## Skill eval: {suite['name']}", "",
             f"**{verdict}** — {'; '.join(reasons) if reasons else 'no regression'}", "",
             f"model `{suite['model']}`, judge `{suite['judge_model']}`, reps {suite['reps']}, "
             f"arms {', '.join(arms)}, spent ${spent:.2f}", "",
             "| case | head | base |", "|---|---|---|"]
    for cid in dict.fromkeys(r["case"] for r in rows):
        cell = lambda rs: "-" if not rs else f"{sum(c['pass'] for r in rs for c in r['criteria'] if c['required'])}/" \
                                             f"{sum(1 for r in rs for c in r['criteria'] if c['required'])}"
        lines.append(f"| {cid} | {cell([r for r in head if r['case'] == cid])} | "
                     f"{cell([r for r in base if r['case'] == cid])} |")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    lines += ["", f"required pass rate: head {fmt(rate(head))}, base {fmt(rate(base))}", "", "### Failed criteria (head)", ""]
    case_by = {}
    for r in head:
        for c in r["criteria"]:
            if c["required"] and not c["pass"]:
                case_by.setdefault((r["case"], c["index"]), []).append(c["reason"])
    lines += [f"- `{cid}` #{i}: {reasons_[0]}" for (cid, i), reasons_ in case_by.items()] or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--base-ref", help="git ref of the previous skill (e.g. origin/main); omit for head only")
    ap.add_argument("--out", default="skill-eval-results.json")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--select", help="comma-separated case ids (overrides suite select)")
    ap.add_argument("--dry-run", action="store_true", help="validate the suite and print the planned call count")
    args = ap.parse_args(argv)

    suite_path = Path(args.suite).resolve()
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               cwd=suite_path.parent).stdout.strip() or ".")
    suite = json.loads(suite_path.read_text("utf-8"))
    suite.setdefault("model", "sonnet")
    suite.setdefault("judge_model", "sonnet")
    suite.setdefault("reps", 2)
    suite.setdefault("jobs", 4)
    suite.setdefault("min_pass_rate", 0.0)
    suite.setdefault("max_drop", 0.1)
    if args.reps:
        suite["reps"] = args.reps
    if args.select:
        suite["select"] = args.select.split(",")
    os.chdir(root)
    cases = load_cases(suite, root)
    checks = load_checks(suite.get("checks"), root)

    systems = {"head": skill_system_prompt(skill_files_worktree(suite["skill_dir"]))}
    if args.base_ref:
        base_files = skill_files_at_ref(suite["skill_dir"], args.base_ref)
        if base_files is not None:
            systems["base"] = skill_system_prompt(base_files)
    jobs = [(arm, case, rep) for arm in systems for case in cases for rep in range(suite["reps"])]
    print(f"{suite['name']}: {len(cases)} cases x {suite['reps']} reps x arms {list(systems)} = "
          f"{len(jobs)} runs ({2 * len(jobs)} claude calls), max_usd {suite['max_usd']}", flush=True)

    runner = Runner(float(suite["max_usd"]), dry_run=args.dry_run)
    if args.dry_run:
        return 0
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=suite["jobs"]) as pool:
        futures = [pool.submit(run_one, runner, suite, checks, arm, systems[arm], case, rep) for arm, case, rep in jobs]
        for fut in futures:
            try:
                row = fut.result()
                rows.append(row)
                print(f"  {row['arm']} {row['case']}#{row['rep']}: "
                      f"{sum(c['pass'] for c in row['criteria'])}/{len(row['criteria'])}", flush=True)
            except (BudgetExceeded, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))

    head_rate = rate([r for r in rows if r["arm"] == "head"])
    base_rate = rate([r for r in rows if r["arm"] == "base"])
    reasons = []
    if errors:
        reasons.append(f"{len(errors)} runs did not finish: {errors[0]}")
    if head_rate is None:
        reasons.append("no head results")
    elif head_rate < suite["min_pass_rate"]:
        reasons.append(f"head pass rate {head_rate:.2f} < min_pass_rate {suite['min_pass_rate']}")
    if head_rate is not None and base_rate is not None and head_rate < base_rate - suite["max_drop"]:
        reasons.append(f"head {head_rate:.2f} dropped more than {suite['max_drop']} below base {base_rate:.2f}")
    verdict = "FAIL" if reasons else "PASS"

    Path(args.out).write_text(json.dumps({
        "suite": suite["name"], "verdict": verdict, "reasons": reasons, "spent_usd": round(runner.spent, 4),
        "model": suite["model"], "judge_model": suite["judge_model"], "reps": suite["reps"],
        "base_ref": args.base_ref if "base" in systems else None,
        "pass_rate": {"head": head_rate, "base": base_rate}, "runs": rows}, ensure_ascii=False, indent=1) + "\n")
    report = summarize(suite, rows, list(systems), runner.spent, verdict, reasons)
    print(report)
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
