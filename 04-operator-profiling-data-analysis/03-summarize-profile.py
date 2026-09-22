#!/usr/bin/env python3
"""Reconcile exported Ascend CSV/JSON files without importing torch or using NPU."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DTYPE_BYTES = {"INT8": 1, "DT_BF16": 2, "FLOAT16": 2, "FLOAT": 4, "INT32": 4, "INT64": 8}
PEAK_INT8 = 1.12e15
PEAK_BF16 = 560e12
PEAK_BANDWIDTH = 3.2e12
MATMUL_TYPES = {"QuantBatchMatmulV3", "MatMulV2", "MatMulV3"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def shapes(value):
    return [tuple(int(v) for v in part.split(",")) if part else ()
            for part in value.strip('"').split(";")]


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def stats(values):
    return {"count": len(values), "total_ms": sum(values) / 1000,
            "mean_us": sum(values) / len(values), "min_us": min(values),
            "p50_us": percentile(values, .5), "p90_us": percentile(values, .9),
            "p99_us": percentile(values, .99), "max_us": max(values)}


def duration_groups(rows, key, duration):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(float(row[duration]))
    return {name: stats(values) for name, values in groups.items()}


def union_us(intervals):
    total, right = 0., -math.inf
    for start, end in sorted(intervals):
        total += max(0., end - max(start, right))
        right = max(right, end)
    return total


def roofline(rows):
    flops, byte_count, lower_us = 0, 0, 0.
    for row in rows:
        ins, outs = shapes(row["Input Shapes"]), shapes(row["Output Shapes"])
        assert len(ins[0]) == len(outs[0]) == 2, row["Input Shapes"]
        m, k = ins[0]
        out_m, n = outs[0]
        assert out_m == m
        assert math.prod(ins[1]) == k * n, (ins, outs)
        work = 2 * m * k * n
        traffic = sum(math.prod(shape) * DTYPE_BYTES[dtype]
                      for shape, dtype in zip(ins, row["Input Data Types"].split(";")))
        traffic += sum(math.prod(shape) * DTYPE_BYTES[dtype]
                       for shape, dtype in zip(outs, row["Output Data Types"].split(";")))
        peak = PEAK_INT8 if row["Type"] == "QuantBatchMatmulV3" else PEAK_BF16
        lower_us += max(work / peak, traffic / PEAK_BANDWIDTH) * 1e6
        flops += work
        byte_count += traffic
    result = stats([float(row["Duration(us)"]) for row in rows])
    seconds = result["total_ms"] / 1000
    result.update({"operations": flops, "modeled_bytes": byte_count,
                   "effective_tops": flops / seconds / 1e12,
                   "effective_tb_s": byte_count / seconds / 1e12,
                   "compute_util_pct": 100 * flops / seconds / peak,
                   "modeled_mbu_pct": 100 * byte_count / seconds / PEAK_BANDWIDTH,
                   "roofline_lower_ms": lower_us / 1000,
                   "ideal_speedup": result["total_ms"] / (lower_us / 1000)})
    for field in ("aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "cube_utilization(%)"):
        result["duration_weighted_" + field] = sum(
            float(row[field]) * float(row["Duration(us)"]) for row in rows
        ) / (seconds * 1e6)
    return result


def summarize(rank_path):
    output = rank_path / "ASCEND_PROFILER_OUTPUT"
    required = ("step_trace_time.csv", "op_statistic.csv", "kernel_details.csv",
                "operator_details.csv", "task_time.csv", "communication.json", "communication_matrix.json")
    files = {}
    for name in required:
        path = output / name
        assert path.is_file() and path.stat().st_size > 0, path
        with path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        files[name] = {"bytes": path.stat().st_size, "sha256": checksum}
    step_rows = read_csv(output / "step_trace_time.csv")
    assert len(step_rows) == 1 and not step_rows[0]["Step"], step_rows
    stage = {name: float(value) / 1000 for name, value in step_rows[0].items()
             if name not in ("Device_id", "Step")}
    assert abs(stage["Stage"] - stage["Computing"] - stage["Communication(Not Overlapped)"] - stage["Free"]) < .001
    assert abs(stage["Communication"] - stage["Communication(Not Overlapped)"] - stage["Overlapped"]) < .001
    kernels = read_csv(output / "kernel_details.csv")
    kernel_groups = duration_groups(kernels, "Type", "Duration(us)")
    op_rows = read_csv(output / "op_statistic.csv")
    cores = defaultdict(float)
    op_types = defaultdict(lambda: {"count": 0, "total_ms": 0., "cores": []})
    for row in op_rows:
        typ = row["OP Type"]
        op_types[typ]["count"] += int(row["Count"])
        op_types[typ]["total_ms"] += float(row["Total Time(us)"]) / 1000
        op_types[typ]["cores"].append(row["Core Type"])
        cores[row["Core Type"]] += float(row["Total Time(us)"]) / 1000
    for typ, data in op_types.items():
        assert data["count"] == kernel_groups[typ]["count"], typ
        assert abs(data["total_ms"] - kernel_groups[typ]["total_ms"]) < .005, typ
        data["calls_per_stage_second"] = data["count"] / (stage["Stage"] / 1000)
    shape_groups = defaultdict(list)
    quant_by_m = defaultdict(list)
    for row in kernels:
        if row["Type"] in MATMUL_TYPES:
            shape_groups[(row["Type"], row["Input Shapes"], row["Output Shapes"])].append(row)
        if row["Type"] == "QuantBatchMatmulV3":
            quant_by_m[shapes(row["Input Shapes"])[0][0]].append(row)
    shape_metrics = []
    for (typ, ins, outs), rows in shape_groups.items():
        m, k = shapes(ins)[0]
        n = shapes(outs)[0][1]
        shape_metrics.append({"type": typ, "m": m, "k": k, "n": n,
                              "input_shapes": ins, **roofline(rows)})
    rooflines = {}
    for typ in sorted(MATMUL_TYPES):
        for bucket in ("all", "M<=128", "M>128"):
            selected = [row for row in kernels if row["Type"] == typ and (
                bucket == "all" or (shapes(row["Input Shapes"])[0][0] <= 128) == (bucket == "M<=128"))]
            if selected:
                rooflines[typ + "/" + bucket] = roofline(selected)
    qkv = [row for row in kernels if row["Type"] == "QuantBatchMatmulV3"
           and shapes(row["Input Shapes"])[0][1] == 5120
           and shapes(row["Output Shapes"])[0][1] == 8192]
    qkv.sort(key=lambda row: float(row["Start Time(us)"]))
    # This checkpoint has 48 linear-attention layers per target forward.
    assert len(qkv) % 48 == 0
    first_time = min(float(row["Start Time(us)"]) for row in kernels)
    forward_anchors = []
    for i in range(0, len(qkv), 48):
        group = qkv[i:i + 48]
        ms = {shapes(row["Input Shapes"])[0][0] for row in group}
        assert len(ms) == 1, ms
        forward_anchors.append({"index": i // 48 + 1, "m": ms.pop(),
                                "anchor_ms_from_first_kernel": (float(group[0]["Start Time(us)"]) - first_time) / 1000})
    host_rows = read_csv(output / "operator_details.csv")
    host_groups = defaultdict(lambda: defaultdict(float))
    for row in host_rows:
        data = host_groups[row["Name"]]
        data["count"] += 1
        for key in ("Host Self Duration(us)", "Host Total Duration(us)", "Device Self Duration(us)", "Device Total Duration(us)"):
            data[key.replace("(us)", "(ms)")] += float(row[key]) / 1000
    task_rows = read_csv(output / "task_time.csv")
    task_groups = duration_groups(task_rows, "kernel_type", "task_time(us)")
    communication = json.loads((output / "communication.json").read_text())
    comm_groups = defaultdict(list)
    comm_totals = defaultdict(float)
    comm_bandwidth = defaultdict(lambda: {"size_mb": 0., "time_ms": 0.})
    comm_intervals = []
    comm_names = Counter()
    for step, contents in communication.items():
        for category in ("collective", "p2p"):
            entries = contents.get(category, {})
            sums = defaultdict(float)
            for name, data in entries.items():
                if name == "Total Op Info":
                    continue
                comm_names[category] += 1
                typ = re.search(r"hcom_(\w+?)_", name).group(1)
                timing = data["Communication Time Info"]
                comm_groups[typ].append(timing)
                for key, value in timing.items():
                    if key.endswith("(ms)"):
                        sums[key] += value
                        comm_totals[key] += value
                start = timing["Start Timestamp(us)"]
                comm_intervals.append((start, start + timing["Elapse Time(ms)"] * 1000))
                for transport, bandwidth in data["Communication Bandwidth Info"].items():
                    comm_bandwidth[transport]["size_mb"] += bandwidth["Transit Size(MB)"]
                    comm_bandwidth[transport]["time_ms"] += bandwidth["Transit Time(ms)"]
            if entries:
                for key, total in sums.items():
                    assert math.isclose(total, entries["Total Op Info"]["Communication Time Info"][key], abs_tol=1e-5)
    comm_by_type = {name: {"count": len(values), **{
        key: sum(value[key] for value in values) for key in comm_totals}}
        for name, values in comm_groups.items()}
    matrix = json.loads((output / "communication_matrix.json").read_text())
    matrix_totals, matrix_extremes = {}, {}
    for step, contents in matrix.items():
        for category in ("collective", "p2p"):
            for name, links in contents.get(category, {}).items():
                target = matrix_totals if "-total@" in name else matrix_extremes
                target[name] = links
    # Shape-compatible neighbours are candidates; CSVs do not prove tensor identity.
    sequences = defaultdict(list)
    for row in kernels:
        if row["Accelerator Core"] != "AI_CPU":
            sequences[row["Stream ID"]].append(row)
    adjacency = Counter()
    adjacency_times = defaultdict(lambda: {"count": 0, "producer_ms": 0., "consumer_ms": 0.})
    for stream in sequences.values():
        stream.sort(key=lambda row: float(row["Start Time(us)"]))
        for previous, row in zip(stream, stream[1:]):
            if previous["Type"] == "DynamicQuant" and row["Type"] == "QuantBatchMatmulV3":
                if shapes(previous["Output Shapes"])[0] == shapes(row["Input Shapes"])[0]:
                    adjacency["DynamicQuant->QuantBatchMatmulV3"] += 1
            if row["Type"] == "DynamicQuant":
                adjacency[previous["Type"] + "->DynamicQuant"] += 1
                if shapes(previous["Output Shapes"])[0] == shapes(row["Input Shapes"])[0]:
                    data = adjacency_times[previous["Type"] + "->DynamicQuant"]
                    data["count"] += 1
                    data["producer_ms"] += float(previous["Duration(us)"]) / 1000
                    data["consumer_ms"] += float(row["Duration(us)"]) / 1000
    return {"rank_directory": str(rank_path), "device_id": step_rows[0]["Device_id"],
            "files": files, "profiler_info": json.loads(next(rank_path.glob("profiler_info_*.json")).read_text()),
            "stage_ms": stage, "op_statistic_rows": len(op_rows), "kernel_rows": len(kernels),
            "op_total_ms": sum(row["total_ms"] for row in op_types.values()), "ops": dict(op_types),
            "cores_ms": dict(cores), "kernels": kernel_groups, "host_ops": dict(host_groups),
            "tasks": task_groups, "roofline": rooflines,
            "shape_metrics": sorted(shape_metrics, key=lambda row: -row["total_ms"]),
            "quant_m": {str(m): stats([float(row["Duration(us)"]) for row in rows]) for m, rows in quant_by_m.items()},
            "forward_anchors": forward_anchors, "target_token_rows_per_layer": sum(row["m"] for row in forward_anchors),
            "communication": {"counts": dict(comm_names), "totals_ms": dict(comm_totals),
                              "by_type": comm_by_type, "bandwidth": dict(comm_bandwidth),
                              "interval_union_ms": union_us(comm_intervals) / 1000},
            "matrix_totals": matrix_totals, "matrix_examples": matrix_extremes,
            "adjacency_candidates": dict(adjacency), "adjacency_times": dict(adjacency_times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["bs16_prefill", "bs16_decode", "bs32_prefill", "bs32_decode"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "vllm_profile"
    result = {"method": {"compute_peak_int8_ops_s": PEAK_INT8, "compute_peak_bf16_ops_s": PEAK_BF16,
                         "bandwidth_bytes_s": PEAK_BANDWIDTH,
                         "peak_scope": "README divided by 8; conditional reference, logical device scope unverified",
                         "memory_model": "Every listed tensor read or written once per invocation; no HBM counters",
                         "small_m_boundary": 128}, "cases": {}}
    for case in args.cases:
        ranks = sorted((root / case).glob("rank*_ascend_pt"))
        assert len(ranks) == 2, (case, ranks)
        result["cases"][case] = {}
        for rank in ranks:
            print(f"Summarizing {case}/{rank.name}", flush=True)
            result["cases"][case][rank.name.split("_")[0]] = summarize(rank)
        a, b = result["cases"][case].values()
        assert set(a["ops"]) == set(b["ops"])
        assert all(a["ops"][op]["count"] == b["ops"][op]["count"] for op in a["ops"])
        assert a["target_token_rows_per_layer"] == b["target_token_rows_per_layer"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
