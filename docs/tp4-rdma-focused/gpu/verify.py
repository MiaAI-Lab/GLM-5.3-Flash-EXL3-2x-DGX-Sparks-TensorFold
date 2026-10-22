#!/usr/bin/env python3
"""Offline, standard-library verification of focused TP4 transport receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


PINNED_SHA256 = {
    "captures/focused-nccl/capture.json": "671ae1d5360ee60ca3ea7b5e11b5f4e55a4f262d448abf3f1118550f6bb963d1",
    "captures/focused-nccl/report.json": "c1c761086c1569bced481845c549b29838206da583a10edd3784bacdae04f656",
    "captures/focused-nccl/report.txt": "04bebb8a4127b40e5682a8140884d17f119c92a860ce2e36a03d10cfb59274f3",
    "captures/focused-nccl/workload.json": "a2eb10e38e7c351c91a3dc78452af3e99b505d718342c3344517375f8567c959",
    "captures/focused-nccl/deployment.json": "bffbd068df2f1b53238dc1b833ef08e546129ae70db9b62223a6c6b0f687b5cf",
    "captures/focused-registered-from-nccl/capture.json": "fa66cf3de7ddc514260af8c53a07a8af155414a0de93686c58ad03e38deff8bc",
    "captures/focused-registered-from-nccl/report.json": "7959708d185fa2ad16f6e1708c40080d886b760ed2868974d647b7fded4cb1f4",
    "captures/focused-registered-from-nccl/report.txt": "212ee85a9c85d40c3a9d4ea2bce8c2d41760b0a41eb0fb3cd6ab3a0b7a4f35c8",
    "captures/focused-registered-from-nccl/workload.json": "a2eb10e38e7c351c91a3dc78452af3e99b505d718342c3344517375f8567c959",
    "captures/focused-registered-from-nccl/deployment.json": "6e3bd6220d297d453ba69b4ddb44ee3310476a6b8871f87e5745aae4f09317a3",
    "final-serving.json": "5092ba5cafa95b31c44eebf3d6460029536e0c0f9b483735eba016d50ca0bb0f",
    "captures/nccl-qualification.jsonl": "a41a811502c80bd0ac9cd3531ddf1a6bc37b04b5776104be29916a88bf6190e9",
    "captures/registered-qualification.jsonl": "a856818e522a2e765a5040e02eefd979f9f0951f39c52cc4f8bc4aaa6de63477",
    "captures/eager-qualification.jsonl": "14df474e158bb394f2f605d0fe1ca30f0840791fa5b37726e730f30865a2bbc2"
}
PINNED_TREES = {
    "captures/focused-nccl": {
        "sha256": "e8d29e32e76f77d447cced085285426da69f78da440785f36ebe4e22cee9c0a0",
        "files": 141
    },
    "captures/focused-registered-from-nccl": {
        "sha256": "004ad64efd7b15a241882764bb2a3a7db96eeb9d233bfb85b48f2b119e70cc05",
        "files": 141
    }
}
CAPTURES = ("focused-nccl", "focused-registered-from-nccl")
PROFILES = ("nccl", "registered", "eager")
MODEL = "GLM-5.3-Flash-EXL3"
CRITICAL_VALUE = 2.364624251
TIMING_FIELDS = {
    "timestamp", "started_at", "finished_at", "start_monotonic_s",
    "end_monotonic_s", "client_ttft_s", "client_elapsed_s",
    "client_decode_tok_s", "client_end_to_end_tok_s",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual), label)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), label)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def sha(path):
    return digest(path.read_bytes())


def canonical_hash(obj):
    return digest(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode())


def verify_capture(root, name):
    base = root / "captures" / name
    capture = read_json(base / "capture.json")
    workload = read_json(base / "workload.json")
    deployment = read_json(base / "deployment.json")
    require(capture["status"] == "complete" and capture["stop_reason"] is None,
            name + ": complete capture")
    require(capture["model"] == MODEL, name + ": model")
    require(capture["metric_contract"] == "generated-text-arrival-v2", name + ": metric")
    require(capture["workload_sha256"] == sha(base / "workload.json"), name + ": workload SHA")
    require(capture["deployment_sha256"] == sha(base / "deployment.json"), name + ": deployment SHA")
    request_spec = workload["request"]
    require(request_spec["output"] == {"tokens": 400, "mode": "exact"}, name + ": exact400")
    require(request_spec["stream"] is True and request_spec["thinking"] is False,
            name + ": request mode")
    require(request_spec["temperature_milli"] == 0 and request_spec["top_p_milli"] == 1000,
            name + ": sampling dimensions")
    require(workload["cells"] == [{"id": "structured-1", "case": "structured",
            "concurrency": 1, "warmup_trials": 1, "trials": 3}], name + ": C1 dimensions")
    acquisitions = capture["acquisitions"]
    require(len(acquisitions) == 8, name + ": acquisition count")
    medians = []
    all_requests = []
    intervals = []
    response_count = 0
    for index, acquisition in enumerate(acquisitions):
        directory = "acquisition-%02d" % index
        require(acquisition["directory"] == directory, name + ": acquisition order")
        require(acquisition["status"] == "completed", name + ": acquisition status")
        started, finished = acquisition["started_unix_ms"], acquisition["finished_unix_ms"]
        require(started < finished, name + ": acquisition interval")
        if intervals:
            require(started >= intervals[-1][1], name + ": sequential acquisitions")
        intervals.append([started, finished])
        acq = base / directory
        plan = read_json(acq / "plan.json")
        plan_sha = sha(acq / "plan.json")
        require(acquisition["plan_sha256"] == plan_sha, name + ": plan SHA")
        require(plan["workload"] == workload and plan["deployment"] == deployment,
                name + ": plan declarations")
        require(plan["source_sha256"] == capture["workload_sha256"], name + ": source SHA")
        require(plan["workload_sha256"] == canonical_hash(workload), name + ": normalized workload SHA")
        require(plan["collector_sha256"] == capture["collector_sha256"], name + ": collector identity")
        require(plan["model"] == MODEL and plan["endpoint"] == capture["endpoint"], name + ": endpoint")
        require(sha(acq / "workload.json") == capture["workload_sha256"], name + ": acquisition workload SHA")
        run = read_json(acq / "run.json")
        require(run == read_json(acq / "session-000000/run.json"), name + ": run copies")
        require(run["status"] == "completed" and run["planned_waves"] == 4
                and run["published_waves"] == 4 and run["measured_waves"] == 3
                and run["eligible_measured_waves"] == 3, name + ": run accounting")
        session = read_json(acq / "session-000000/session.json")
        require(session["plan_sha256"] == plan_sha
                and session["collector_sha256"] == capture["collector_sha256"], name + ": session SHA links")
        require(len(plan["waves"]) == 4, name + ": wave count")
        measured = []
        for wave_index, spec in enumerate(plan["waves"]):
            require(spec == {"index": wave_index, "phase": "warmup" if wave_index == 0 else "measured",
                    "cell": "structured-1", "case": "structured", "trial": max(0, wave_index - 1),
                    "concurrency": 1}, name + ": wave dimensions")
            wave_dir = acq / ("wave-%06d" % wave_index)
            wave = read_json(wave_dir / "wave.json")
            reservation = read_json(wave_dir / "reservation.json")
            require(wave["plan_sha256"] == reservation["plan_sha256"] == plan_sha, name + ": wave plan SHA")
            require(wave["reservation_sha256"] == sha(wave_dir / "reservation.json"), name + ": reservation SHA")
            require(wave["spec"] == reservation["wave"] == spec, name + ": wave spec")
            require(len(reservation["requests"]) == len(reservation["request_sha256"]) == 1,
                    name + ": C1 request count")
            request = reservation["requests"][0]
            require(digest(request.encode()) == reservation["request_sha256"][0], name + ": request SHA")
            decoded = json.loads(request)
            require(decoded == {"model": MODEL, "messages": workload["cases"][0]["messages"],
                    "stream": True, "max_tokens": 400, "temperature": 0.0, "top_p": 1.0,
                    "chat_template_kwargs": {"thinking": False}, "stream_options": {"include_usage": True},
                    "min_tokens": 400, "ignore_eos": True}, name + ": request dimensions")
            all_requests.append(request)
            require(wave["eligible"] is True and len(wave["attempts"]) == 1, name + ": wave eligibility")
            attempt = wave["attempts"][0]
            require(attempt["dispatched"] is True and attempt["status"] == "complete"
                    and attempt["http_status"] == 200 and attempt["finish_reason"] == "length"
                    and attempt["eligibility_errors"] == [], name + ": request completion")
            require(attempt["usage"]["completion_tokens"] == wave["completion_tokens"] == 400,
                    name + ": output dimensions")
            raw = (wave_dir / "response-0000.bin").read_bytes()
            require(digest(raw) == attempt["response_sha256"] and len(raw) == attempt["response_bytes"],
                    name + ": response SHA/length")
            require(attempt["terminal_offset"] == len(raw) and attempt["surplus_observed_bytes"] == 0,
                    name + ": response terminal dimensions")
            data = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
            require(data and data[-1] == "[DONE]", name + ": SSE terminal")
            events = [json.loads(line) for line in data[:-1]]
            usages = [event["usage"] for event in events if event.get("usage")]
            require(usages and usages[-1]["completion_tokens"] == 400
                    and usages[-1]["prompt_tokens"] == attempt["usage"]["prompt_tokens"], name + ": SSE usage")
            require(any(choice.get("finish_reason") == "length"
                    for event in events for choice in event.get("choices", [])), name + ": SSE finish")
            near(wave["achieved_completion_tokens_per_second"],
                 400e6 / wave["elapsed_us"], name + ": wave throughput")
            if wave_index:
                measured.append(wave["achieved_completion_tokens_per_second"])
            response_count += 1
        median = statistics.median(measured)
        near(acquisition["median_achieved_completion_tokens_per_second"], median, name + ": acquisition median")
        medians.append(median)
    require(len(set(all_requests)) == 1, name + ": identical fixed request bytes")
    require(capture["request_ceiling"] == 32 and capture["output_token_ceiling"] == 12800,
            name + ": capture ceilings")
    return {"capture_sha256": sha(base / "capture.json"), "report_sha256": sha(base / "report.json"),
            "acquisitions": 8, "requests": response_count, "measured_requests": 24,
            "reported_completion_tokens": 12800, "acquisition_medians_tok_s": medians,
            "median_acquisition_tok_s": statistics.median(medians), "interval_unix_ms": [intervals[0][0], intervals[-1][1]],
            "request_sha256": digest(all_requests[0].encode())}


def verify_qualifications(root):
    outputs = {}
    signatures = {}
    for profile in PROFILES:
        path = root / "captures" / (profile + "-qualification.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == 89 and all("backend" not in row for row in rows), profile + ": projected rows")
        require(all(row.get("status", "passed") == "passed" for row in rows), profile + ": statuses")
        require(rows[0]["model"] == MODEL and rows[0]["suite"] == "qualification", profile + ": suite")
        summary = rows[-1]
        require(summary["case"] == "summary" and summary["attempted_http_jobs"] ==
                {"models": 2, "tokenizer": 31, "completion": 40}, profile + ": fixture accounting")
        completions = [row for row in rows if row.get("kind") == "completion"]
        require(len(completions) == 40, profile + ": exact40 completions")
        keys = [(row["case"], row["key"]) for row in completions]
        require(len(set(keys)) == 40, profile + ": unique fixture identities")
        tokens_by_case = {}
        for row in completions:
            label = profile + ": " + row["case"]
            require(digest(row["text"].encode()) == row["text_sha256"], label + " text SHA")
            token_ids = row["tensorfold"]["token_ids"]
            require(len(token_ids) == row["usage"]["completion_tokens"], label + " token count")
            token_sha = digest(",".join(map(str, token_ids)).encode())[:12]
            require(token_sha == row["tensorfold"]["token_sha"], label + " token SHA")
            require(row["requested_draft"] == row["tensorfold"]["drafts"]
                    and row["draft_enabled_verified"] == row["requested_draft"], label + " draft dimension")
            require(row["tensorfold"]["enable_thinking"] is False, label + " thinking dimension")
            tokens_by_case[(row["case"], row["key"])] = token_ids
        boundary_lengths = [1023, 1024, 1025, 1535, 1536, 1537, 2050, 2051, 2052]
        for length in boundary_lengths:
            serial = next(row for row in completions if row["case"] == "boundary-%d-serial-1" % length)
            draft = next(row for row in completions if row["case"] == "boundary-%d-draft" % length)
            require(serial["usage"]["prompt_tokens"] == draft["usage"]["prompt_tokens"] == length,
                    profile + ": boundary dimensions")
            require(serial["tensorfold"]["token_ids"] == draft["tensorfold"]["token_ids"],
                    profile + ": serial/draft tokens")
        require([tokens_by_case[("solo-%d" % i, "hermes")] for i in range(8)] ==
                [tokens_by_case[("overlap-%d" % i, "hermes")] for i in range(8)], profile + ": overlap tokens")
        signatures[profile] = [{key: value for key, value in row.items() if key not in TIMING_FIELDS}
                               for row in completions]
        outputs[profile] = {"sha256": sha(path), "rows": len(rows), "completions": 40,
                             "all_passed": True, "token_and_text_hashes_checked": 40}
    require(signatures["nccl"] == signatures["registered"] == signatures["eager"],
            "exact40 completion output and all non-timing field parity")
    return {"profiles": outputs, "completion_output_and_dimensions_parity": True,
            "rendered_prompt_boundaries": boundary_lengths, "serial_draft_parity": True,
            "solo_overlap_parity": True, "body_and_seed_verification": "unavailable in qualification receipts"}


def verify_final_serving(root):
    """Recheck current candidate acceptance without executing its collector."""
    report = read_json(root / "final-serving.json")
    require(report["ok"] is True and report["schema"] == 1, "final serving report")
    checked_sources = 0

    def check_source(source):
        nonlocal checked_sources
        declared = source["path"]
        relative = "captures/" + declared.split("/captures/", 1)[1] if "/captures/" in declared else Path(declared).name
        path = root / relative
        require(sha(path) == source["sha256"] and path.stat().st_size == source["bytes"],
                "final serving source SHA/length: " + relative)
        checked_sources += 1
        return read_json(path) if path.suffix == ".json" else None

    for field in ("final_candidate_ids", "eager_graph_mode"):
        require(check_source(report[field]["source"]) == report[field]["receipt"], field + ": receipt equality")
    ids = report["final_candidate_ids"]["receipt"]
    require(set(ids) == {"spark", "spark-peer", "msi-b9dc", "msi-c21f"}
            and len(set(ids.values())) == 4 and all(len(value) == 64 for value in ids.values()), "final candidate IDs")
    require(report["eager_graph_mode"]["verified"] is True
            and report["eager_graph_mode"]["receipt"]["passed"] is True, "eager/graph configuration receipt")
    check_source(report["serving_console"]["source"])
    supplemental = report["supplemental_acceptance"]
    require(supplemental["ok"] is True and not supplemental["validation_errors"], "supplemental report")
    for source in supplemental["sources"]:
        check_source(source)
    base = root / "captures/accept-focused-final"
    acceptance = read_json(base / "report.json")
    require(acceptance["ok"] is True and acceptance["model"] == MODEL, "acceptance report")
    require(supplemental["case_count"] == len(acceptance["settled_requests"]) == 31, "acceptance case count")
    results = {}
    for settled in acceptance["settled_requests"]:
        label = settled["label"]
        require(label not in results and "/" not in label, "acceptance identity")
        record = read_json(base / (label + "-result.json"))
        snapshot = read_json(base / (label + "-request.json"))
        require(record["ok"] is True and settled["ok"] is True and record["http_status"] == 200,
                label + ": settled result")
        require(record["request"] == snapshot["request"], label + ": request snapshot")
        encoded = json.dumps({"path": record["path"], "body": record["request"]}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        require(digest(encoded) == record["request_sha256"] == settled["request_sha256"] == snapshot["request_sha256"],
                label + ": request SHA")
        raw = (base / (label + "-response.bin")).read_bytes()
        require(digest(raw) == record["response_sha256"] == settled["response_sha256"], label + ": response SHA")
        result = record["result"]
        if record["path"] == "/tokenize":
            require(json.loads(raw) == result and result["count"] == len(result["tokens"]), label + ": tokenizer response")
        else:
            lines = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
            events = [json.loads(line) for line in lines if line != "[DONE]"]
            if label == "cancel-essay":
                require(result["closed_after_first_content"] is True and result["complete_stream"] is False
                        and events == result["events_before_close"], label + ": closed stream receipt")
            else:
                require(lines[-1] == "[DONE]" and events == result["sse_events"], label + ": SSE receipt")
                token_events = [event["tensorfold"] for event in events if event.get("tensorfold", {}).get("token_ids")]
                require(len(token_events) == 1, label + ": token event")
                tokens = token_events[0]["token_ids"]
                require(tokens == result["token_ids"] and len(tokens) == result["usage"]["completion_tokens"],
                        label + ": token dimensions")
                require(digest(json.dumps(tokens, separators=(",", ":")).encode()) == result["token_ids_sha256"],
                        label + ": token IDs SHA")
                require(record["request"]["draft"] == token_events[0]["drafts"], label + ": draft dimension")
        results[label] = record
    comparisons = acceptance["comparisons"]
    require(len(comparisons) == supplemental["pair_comparison_count"] == 12, "exact12 pairs")
    for pair in comparisons:
        left, right = [results[pair[side]] for side in ("left", "right")]
        left_request, right_request = dict(left["request"]), dict(right["request"])
        if pair["ignore_draft_only"]:
            require(left_request.pop("draft") is False and right_request.pop("draft") is True, "pair draft selector")
        require(left_request == right_request, "pair request dimensions")
        require(pair["exact_match"] is True and left["result"]["token_ids"] == right["result"]["token_ids"]
                and left["result"]["text"] == right["result"]["text"], "pair exact output")
        require(left["result"]["token_ids_sha256"] == right["result"]["token_ids_sha256"] == pair["token_ids_sha256"],
                "pair token SHA")
    cancel = results["cancel-essay"]
    require(all(results[label]["started_monotonic_s"] > cancel["settled_monotonic_s"]
                for label in ("after-cancel-serial", "after-cancel-draft")), "cancellation follow-up ordering")
    for label in ("needle-serial", "needle-draft", "needle-reuse-1", "needle-reuse-2"):
        require(acceptance["needle"]["required_value"] in results[label]["result"]["text"], "needle value presence")
    return {"sha256": sha(root / "final-serving.json"), "current_candidate_ids": ids,
            "source_hashes_checked": checked_sources, "settled_requests": 31, "exact_output_pairs": 12,
            "request_response_hashes_verified": True, "client_cancellation_followup_verified": True,
            "needle_presence_verified": True,
            "limits": ["Final serving collector and analyzer source provenance are retained as hashes; their source is not executed or reverified here.",
                       "Sustained-load figures remain in the retained report and are not independently recomputed by this validator.",
                       "Needle presence and output parity do not establish clean answer quality.",
                       "Client connection closure and follow-up usability do not independently prove backend cancellation retirement."]}


def verify(root):
    for relative, expected in PINNED_TREES.items():
        base = root / relative
        files = sorted(path for path in base.rglob("*") if path.is_file())
        tree = hashlib.sha256()
        for path in files:
            tree.update(path.relative_to(base).as_posix().encode() + b"\0" + sha(path).encode() + b"\n")
        require(len(files) == expected["files"] and tree.hexdigest() == expected["sha256"],
                "published capture tree SHA: " + relative)
    for relative, expected in PINNED_SHA256.items():
        require(sha(root / relative) == expected, "published receipt SHA: " + relative)
    nccl, registered = [verify_capture(root, name) for name in CAPTURES]
    require(nccl["request_sha256"] == registered["request_sha256"], "transport request parity")
    require(nccl["interval_unix_ms"][1] < registered["interval_unix_ms"][0], "sequential transport capture periods")
    a, b = nccl["acquisition_medians_tok_s"], registered["acquisition_medians_tok_s"]
    la, lb = [math.log(x) for x in a], [math.log(x) for x in b]
    difference = statistics.mean(lb) - statistics.mean(la)
    se = math.sqrt(statistics.variance(la) / 8 + statistics.variance(lb) / 8)
    effect = 100 * math.expm1(difference)
    interval = [100 * math.expm1(difference + sign * CRITICAL_VALUE * se) for sign in (-1, 1)]
    report = read_json(root / "captures" / CAPTURES[1] / "report.json")
    require(report["result"] == "IMPROVED" and report["declared_change"] == "settings", "official result")
    require(report["baseline_capture_sha256"] == nccl["capture_sha256"]
            and report["candidate_capture_sha256"] == registered["capture_sha256"], "report capture SHA links")
    require(report["baseline_acquisition_medians"] == a and report["candidate_acquisition_medians"] == b,
            "report acquisition arrays")
    require(report["baseline_complete_acquisitions"] == report["candidate_complete_acquisitions"] == 8,
            "report acquisition counts")
    for label in ("baseline", "candidate"):
        account = report[label + "_accounting"]
        require(account["dispatched_requests"] == account["complete_requests"] == account["request_ceiling"] == 32
                and account["reported_completion_tokens"] == account["known_reported_completion_tokens"] ==
                account["output_token_ceiling"] == 12800 and account["missing_completion_usage_requests"] == 0,
                "report request accounting")
    near(report["observed_change_percent"], effect, "official effect")
    near(report["log_mean_difference"], difference, "official log difference")
    near(report["heteroscedastic_standard_error"], se, "official standard error")
    require(report["critical_value"] == CRITICAL_VALUE and report["conservative_degrees_of_freedom"] == 7
            and report["model_based_confidence_level"] == 0.95, "official CI dimensions")
    for actual, expected in zip(report["model_based_interval_percent"], interval):
        near(actual, expected, "official interval")
    return {"schema_version": 1, "status": "passed", "method": "offline standard-library receipt verification",
            "published_receipt_hashes_checked": len(PINNED_SHA256),
            "published_capture_tree_hashes": PINNED_TREES,
            "capture_comparison": {"nccl": nccl, "registered_rdma": registered,
                "official_observed_change_percent": effect, "nominal_95_percent_interval": interval,
                "statistic": "100 * expm1(mean(log(candidate acquisition medians)) - mean(log(baseline acquisition medians)))",
                "interval_method": "log-scale heteroscedastic standard error; t critical 2.364624251; conservative df 7",
                "declared_change": "settings"},
            "qualification": verify_qualifications(root),
            "final_serving": verify_final_serving(root),
            "proof_limits": [
                "One short synthetic structured C1/exact400 workload with eight acquisitions per transport.",
                "Sequential capture periods can have drift, carryover and correlated variation; the nominal interval can be too narrow.",
                "No causal attribution, equivalence, noninferiority, guaranteed precision or general serving-performance claim.",
                "Deployment identities and settings are declarations, not runtime attestation.",
                "Qualification receipts omit request bodies and seed; exact request replay cannot be reconstructed from prompt hashes alone.",
                "Rendered prompt boundary coverage does not attest internal prefill widths; client overlap does not attest backend overlap.",
                "No live, GPU, network, build, full-window quality, long-duration reliability or reboot recovery qualification was performed.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True, help="GPU receipt directory")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report")
    args = parser.parse_args()
    try:
        result = verify(args.receipts)
        code = 0
    except (ValueError, KeyError, IndexError, TypeError, OSError, StopIteration) as error:
        result = {"schema_version": 1, "status": "failed", "error": str(error)}
        code = 1
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return code


if __name__ == "__main__":
    sys.exit(main())
