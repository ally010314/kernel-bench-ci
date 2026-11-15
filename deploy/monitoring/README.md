# Monitoring: MLflow + Grafana

Every CI run is recorded in MLflow as one run (`<device>@<sha>`), and Grafana reads MLflow's Postgres
directly to show per-commit trends.

```
CI (Jetson / RTX runner) ──REST──▶ MLflow :5000 ──▶ Postgres ◀── Grafana :3000
```

## Server setup

```bash
cd deploy/monitoring
cp .env.example .env        # set passwords
docker compose up -d --build
```

Then register the tracking URI as a repo secret so CI can log to it:

```bash
gh secret set MLFLOW_TRACKING_URI -R <org>/kernel-bench-ci --body "http://<server-ip>:5000"
```

If you reach MLflow by hostname instead of a private IP, add it to `MLFLOW_ALLOWED_HOSTS` in `.env`
(MLflow 3 only accepts localhost and private IPs by default).

## What gets logged

| kind | keys |
|---|---|
| tags | `sha`, `device`, `gpu`, `branch`, `event`, `run_url` |
| params | `perf_tol`, `gpu_clock_mhz`, `clock_locked` |
| metrics | `fused_ms/N{n}`, `dense_ms/N{n}`, `speedup/N{n}`, `delta/N{n}`, `max_err/N{n}`, `mean_err/N{n}` |

Metric timestamps are the **commit time**, so charts are ordered by commit. Run status is `FAILED` if the
accuracy or perf gate failed.

The client (`bench/mlflow_log.py`) only needs `requests`; the runner does not install MLflow.
Logging is `continue-on-error`, so an MLflow outage never changes the CI verdict.

## Dashboard

`grafana/dashboards/kernel-bench.json` is provisioned automatically (folder *kernel-bench*):
latency trend, speedup vs cuBLAS dense, Δ vs baseline (5% line), max error (5e-3 line), and a recent-runs
table linking to each Actions run. Filter by `device` and `branch`.

Grafana connects with a read-only Postgres role (`grafana`), created on first DB init by
`grafana/readonly-user.sh`.
