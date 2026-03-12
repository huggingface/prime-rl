import wandb
import pandas as pd
import numpy as np
import json
from scipy.stats import linregress
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orch-run", type=str, default="huggingface/prime-rl-bench/sxittc3q", help="WandB run path for orchestrator"
    )
    parser.add_argument(
        "--train-run", type=str, default="huggingface/prime-rl-bench/0z81b95g", help="WandB run path for trainer"
    )
    parser.add_argument("--output-prefix", type=str, default="prime_rl_metrics", help="Prefix for output files")
    args = parser.parse_args()

    api = wandb.Api()

    print(f"Fetching history for orchestrator run: {args.orch_run}...")
    orch_run = api.run(args.orch_run)
    orch_history = orch_run.history(samples=100000, pandas=True)
    orch_system = orch_run.history(stream="events", pandas=True)

    print(f"Fetching history for trainer run: {args.train_run}...")
    train_run = api.run(args.train_run)
    train_history = train_run.history(samples=100000, pandas=True)
    train_system = train_run.history(stream="events", pandas=True)

    # Clean data
    orch_df = orch_history
    train_df = train_history

    # T1: End-to-end tokens/second
    # Prime-RL: Total generated tokens + Total trained tokens / max runtime
    max_runtime = max(orch_df.get("_runtime", pd.Series([0])).max(), train_df.get("_runtime", pd.Series([0])).max())

    total_gen_tokens = orch_df.get("progress/total_tokens", pd.Series([0])).max()
    if "perf/throughput" in train_df and "time/step" in train_df:
        total_train_tokens = (train_df["perf/throughput"] * train_df["time/step"]).sum()
    else:
        total_train_tokens = 0

    t1_e2e_throughput = (total_gen_tokens + total_train_tokens) / max_runtime if max_runtime > 0 else 0

    # L2: Reward trend
    if "reward/all/mean" in orch_df.columns and "step" in orch_df.columns:
        df_rewards = orch_df.dropna(subset=["reward/all/mean", "step"])
        if len(df_rewards) > 1:
            slope, _, _, _, _ = linregress(df_rewards["step"], df_rewards["reward/all/mean"])
        else:
            slope = None
    else:
        slope = None

    # ---- System Metrics Processing ----
    def get_system_metrics(sys_df):
        gpu_util_cols = [c for c in sys_df.columns if "system.gpu." in c and ".gpu/l:" in c]
        gpu_mem_cols = [c for c in sys_df.columns if "system.gpu." in c and ".memoryAllocatedBytes/l:" in c]
        cpu_mem_cols = [c for c in sys_df.columns if "system.memory_percent/l:" in c]
        return gpu_util_cols, gpu_mem_cols, cpu_mem_cols

    orch_util, orch_mem, orch_cpu = get_system_metrics(orch_system)
    train_util, train_mem, train_cpu = get_system_metrics(train_system)

    # U1: GPU Utilization
    u1_util_vals = []
    if orch_util:
        u1_util_vals.extend(orch_system[orch_util].mean().values)
    if train_util:
        u1_util_vals.extend(train_system[train_util].mean().values)
    u1_gpu_util_mean = np.mean(u1_util_vals) if u1_util_vals else None

    # U3: GPU Memory Peak (GB)
    u3_gpu_mem_peak_gb = train_df.get("perf/peak_memory", pd.Series([None])).max()
    if pd.isna(u3_gpu_mem_peak_gb) and train_mem:
        u3_gpu_mem_peak_gb = train_system[train_mem].max().max() / (1024**3)

    # U4: GPU Idle Time
    # Combine system metrics or take max idle time
    idle_time_pcts = []
    if orch_util:
        idle_time_pcts.append((orch_system[orch_util] <= 1.0).any(axis=1).mean() * 100)
    if train_util:
        idle_time_pcts.append((train_system[train_util] <= 1.0).any(axis=1).mean() * 100)
    u4_gpu_idle_time_pct = np.mean(idle_time_pcts) if idle_time_pcts else None

    # R1: CPU Memory Usage Peak (%)
    cpu_peaks = []
    if orch_cpu:
        cpu_peaks.append(orch_system[orch_cpu].max().max())
    if train_cpu:
        cpu_peaks.append(train_system[train_cpu].max().max())
    r1_cpu_mem_peak_pct = max(cpu_peaks) if cpu_peaks else None

    # T6: Wall clock time for 100 steps
    t6_wall_clock = None
    if "step" in train_df.columns and "_runtime" in train_df.columns:
        step_100_rows = train_df[train_df["step"] >= 100]
        if not step_100_rows.empty:
            t6_wall_clock = step_100_rows.iloc[0]["_runtime"]
        else:
            t6_wall_clock = train_df["_runtime"].dropna().iloc[-1] if len(train_df["_runtime"].dropna()) > 0 else None

    # A3: Pipeline bubble
    # Trainer wait time ratio
    trainer_wait_ratio = None
    if "time/wait_for_batch" in train_df.columns and "time/step" in train_df.columns:
        valid_steps = train_df.dropna(subset=["time/wait_for_batch", "time/step"])
        if len(valid_steps) > 0:
            trainer_wait_ratio = (valid_steps["time/wait_for_batch"] / valid_steps["time/step"]).mean()

    # Rollout time
    m2_rollout_time = None
    if "generation_ms/all/mean" in orch_df.columns and "scoring_ms/all/mean" in orch_df.columns:
        m2_rollout_time = ((orch_df["generation_ms/all/mean"] + orch_df["scoring_ms/all/mean"]) / 1000).mean()

    m4_env_latency = None
    if "scoring_ms/all/mean" in orch_df.columns and "num_turns/all/mean" in orch_df.columns:
        m4_env_latency = (orch_df["scoring_ms/all/mean"] / orch_df["num_turns/all/mean"]).mean()

    summary = {
        # 4.1 Throughput
        "T1_e2e_tokens_per_sec_mean": t1_e2e_throughput,
        "T2_train_tokens_per_sec_mean": train_df.get("perf/throughput", pd.Series(dtype=float)).mean(),
        "T3_gen_tokens_per_sec_mean": orch_df.get("perf/throughput", pd.Series(dtype=float)).mean(),
        "T5_steps_per_hour_mean": 3600 / train_df["time/step"].mean() if "time/step" in train_df.columns else None,
        "T6_wall_clock_time_100_steps": t6_wall_clock,
        # 4.2 Hardware Utilization
        "U1_gpu_utilization_mean_pct": u1_gpu_util_mean,
        "U2_mfu_mean_pct": train_df.get("perf/mfu", pd.Series(dtype=float)).mean(),
        "U3_gpu_memory_peak_gb": u3_gpu_mem_peak_gb,
        "U4_gpu_idle_time_pct": u4_gpu_idle_time_pct,
        # 4.3 Async Pipeline Efficiency
        "A1_weight_sync_latency_sec_mean": orch_df.get("time/update_weights", pd.Series(dtype=float)).mean(),
        "A3_pipeline_bubble_ratio_mean": trainer_wait_ratio,
        "A4_policy_staleness_mean": orch_df.get("off_policy_level/all/mean", pd.Series(dtype=float)).mean(),
        "A5_policy_staleness_max": orch_df.get("off_policy_level/all/max", pd.Series(dtype=float)).max(),
        # 4.4 Multi-turn & Straggler
        "M1_turns_per_rollout_mean": orch_df.get("num_turns/all/mean", pd.Series(dtype=float)).mean(),
        "M2_rollout_time_sec_mean": m2_rollout_time,
        "M4_env_latency_ms_per_turn": m4_env_latency,
        # 4.5 Learning Sanity Checks
        "L1_reward_mean": orch_df.get("reward/all/mean", pd.Series(dtype=float)).mean(),
        "L1_reward_std": orch_df.get("reward/all/mean", pd.Series(dtype=float)).std(),
        "L2_reward_trend_slope": slope,
        "L3_kl_divergence_mean": train_df.get("mismatch_kl/mean", pd.Series(dtype=float)).mean(),
        "L4_is_ratio_bound_mean": train_df.get("is_masked/mean", pd.Series(dtype=float)).mean(),
        "L4_is_ratio_bound_max": train_df.get("is_masked/max", pd.Series(dtype=float)).max(),
        # 4.6 Resource
        "R1_cpu_memory_peak_pct": r1_cpu_mem_peak_pct,
    }

    # Save outputs
    summary_file = f"{args.output_prefix}_summary.json"
    with open(summary_file, "w") as f:
        # Convert pandas/numpy types to native Python types for JSON
        def convert(obj):
            if isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj

        json.dump({k: convert(v) for k, v in summary.items()}, f, indent=4)
    print(f"Saved aggregated summary to {summary_file}")

    # Generate Markdown Report
    markdown = f"""# prime-rl Benchmark Metrics Report

## 4.1 Throughput

| Metric | Value | Description |
|---|---|---|
| **T1: End-to-end tokens/second (Mean)** | `{summary.get("T1_e2e_tokens_per_sec_mean") or 0:.2f}` tok/s | Total tokens (generated + trained) / total wall-clock time |
| **T2: Training tokens/second (Mean)** | `{summary.get("T2_train_tokens_per_sec_mean") or 0:.2f}` tok/s | Tokens consumed by gradient steps per second |
| **T3: Generation tokens/second (Mean)** | `{summary.get("T3_gen_tokens_per_sec_mean") or 0:.2f}` tok/s | Tokens produced by inference engine per second |
| **T5: Training steps/hour (Mean)** | `{summary.get("T5_steps_per_hour_mean") or 0:.2f}` steps/h | Gradient update steps completed per hour |
| **T6: Total wall-clock time for 100 steps** | `{summary.get("T6_wall_clock_time_100_steps") or 0:.2f}` seconds | Bottom-line runtime for the benchmark |

## 4.2 Hardware Utilization

| Metric | Value | Description |
|---|---|---|
| **U1: GPU utilization (Mean)** | `{summary.get("U1_gpu_utilization_mean_pct") or 0:.2f}` % | Average SM utilization across all GPUs |
| **U2: MFU (Mean)** | `{summary.get("U2_mfu_mean_pct") or 0:.2f}` % | Model FLOPS Utilization |
| **U3: GPU memory peak** | `{summary.get("U3_gpu_memory_peak_gb") or 0:.2f}` GB | Max allocated memory observed on any single GPU |
| **U4: GPU idle time** | `{summary.get("U4_gpu_idle_time_pct") or 0:.2f}` % | Fraction of wall-clock where at least one GPU has <= 1% SM utilization |

## 4.3 Async Pipeline Efficiency

| Metric | Value | Description |
|---|---|---|
| **A1: Weight sync latency (Mean)** | `{(summary.get("A1_weight_sync_latency_sec_mean") or 0) * 1000:.4f}` ms | Wall-clock time to broadcast new weights to inference engine |
| **A3: Pipeline bubble (Mean)** | `{(summary.get("A3_pipeline_bubble_ratio_mean") or 0) * 100:.2f}` % | Time spent in synchronization barriers / waiting |
| **A4: Average policy staleness** | `{summary.get("A4_policy_staleness_mean") or 0:.2f}` steps | Mean off-policy level |
| **A5: Max policy staleness** | `{summary.get("A5_policy_staleness_max") or 0:.2f}` steps | Max off-policy level |

## 4.4 Multi-turn & Straggler

| Metric | Value | Description |
|---|---|---|
| **M1: Turns per rollout (Mean)** | `{summary.get("M1_turns_per_rollout_mean") or 0:.2f}` turns | Number of model-environment exchanges |
| **M2: Rollout completion time (Mean)** | `{summary.get("M2_rollout_time_sec_mean") or 0:.2f}` seconds | Wall-clock from prompt dispatch to final reward received |
| **M4: Environment latency (Mean)** | `{summary.get("M4_env_latency_ms_per_turn") or 0:.2f}` ms | Sandbox execution time per turn |

## 4.5 Learning Sanity Checks

| Metric | Value | Description |
|---|---|---|
| **L1: Reward (Mean)** | `{summary.get("L1_reward_mean") or 0:.6f}` | Average reward across all rollouts per step |
| **L1: Reward (Std Dev)** | `{summary.get("L1_reward_std") or 0:.6f}` | Standard deviation of reward per step |
| **L2: Reward trend (Slope)** | `{summary.get("L2_reward_trend_slope") or 0:.2e}` | Linear regression slope of reward over training steps |
| **L3: KL divergence (Mean)** | `{summary.get("L3_kl_divergence_mean") or 0:.6f}` nats | Between current policy and behavior policy |
| **L4: IS Ratio Bound (Mean)** | `{summary.get("L4_is_ratio_bound_mean") or 0:.4f}` | Mean importance sampling mask ratio |
| **L4: IS Ratio Bound (Max)** | `{summary.get("L4_is_ratio_bound_max") or 0:.4f}` | Max importance sampling mask ratio |

## 4.6 Resource

| Metric | Value | Description |
|---|---|---|
| **R1: CPU memory usage peak** | `{summary.get("R1_cpu_memory_peak_pct") or 0:.2f}` % | Peak host RAM usage percentage |
"""

    report_file = f"{args.output_prefix}_report.md"
    with open(report_file, "w") as f:
        f.write(markdown)
    print(f"Saved markdown report to {report_file}")


if __name__ == "__main__":
    main()
