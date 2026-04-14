"use client";

import { useState } from "react";
import { runBenchmark } from "@/lib/api";
import type { BackendType, BenchmarkEntry, BenchmarkResult, ModelName } from "@/types";
import { Card, Spinner } from "@/components/ui";

const ALL_MODELS: ModelName[] = ["yolov8", "yolov5"];
const ALL_BACKENDS: BackendType[] = ["pytorch", "torchscript", "onnx"];

const MODEL_LABELS: Record<ModelName, string> = {
  yolov8: "YOLOv8",
  yolov5: "YOLOv5",
};
const BACKEND_LABELS: Record<BackendType, string> = {
  pytorch: "PyTorch",
  torchscript: "TorchScript",
  onnx: "ONNX Runtime",
};

type BenchmarkStatus = "idle" | "running" | "done" | "error";

export default function BenchmarkPanel() {
  const [models, setModels] = useState<Set<ModelName>>(new Set(["yolov8"]));
  const [backends, setBackends] = useState<Set<BackendType>>(new Set(["pytorch"]));
  const [numRuns, setNumRuns] = useState(50);
  const [warmupRuns, setWarmupRuns] = useState(5);
  const [imageSize, setImageSize] = useState(640);
  const [status, setStatus] = useState<BenchmarkStatus>("idle");
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggleModel = (m: ModelName) => {
    setModels((prev) => {
      const next = new Set(prev);
      next.has(m) ? next.delete(m) : next.add(m);
      return next.size > 0 ? next : prev; // always keep at least one
    });
  };

  const toggleBackend = (b: BackendType) => {
    setBackends((prev) => {
      const next = new Set(prev);
      next.has(b) ? next.delete(b) : next.add(b);
      return next.size > 0 ? next : prev;
    });
  };

  const handleRun = async () => {
    setStatus("running");
    setResult(null);
    setError(null);
    try {
      const r = await runBenchmark(
        [...models],
        [...backends],
        numRuns,
        imageSize,
        warmupRuns
      );
      setResult(r);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed.");
      setStatus("error");
    }
  };

  const isRunning = status === "running";
  const canRun = models.size > 0 && backends.size > 0 && !isRunning;

  // Determine the best (lowest avg latency) among successful results
  const okResults = result?.results.filter((r) => r.status === "ok") ?? [];
  const minLatency = okResults.length
    ? Math.min(...okResults.map((r) => r.avg_latency_ms))
    : 0;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-base font-semibold text-slate-200">Inference Benchmark</h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Compare latency and FPS across models and backends using synthetic images.
          TorchScript and ONNX backends require models to be exported first.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-5">
        {/* ── Config sidebar ── */}
        <div className="space-y-4">
          <Card title="Models">
            <div className="space-y-2">
              {ALL_MODELS.map((m) => (
                <CheckCard
                  key={m}
                  label={MODEL_LABELS[m]}
                  checked={models.has(m)}
                  onChange={() => toggleModel(m)}
                  disabled={isRunning}
                />
              ))}
            </div>
          </Card>

          <Card title="Backends">
            <div className="space-y-2">
              {ALL_BACKENDS.map((b) => (
                <CheckCard
                  key={b}
                  label={BACKEND_LABELS[b]}
                  sublabel={
                    b === "torchscript"
                      ? "Requires export step"
                      : b === "onnx"
                      ? "CPU / CUDA provider"
                      : "Baseline"
                  }
                  checked={backends.has(b)}
                  onChange={() => toggleBackend(b)}
                  disabled={isRunning}
                />
              ))}
            </div>
          </Card>

          <Card title="Parameters">
            <div className="space-y-3">
              <NumberInput
                label="Timed Runs"
                value={numRuns}
                min={10}
                max={500}
                step={10}
                onChange={setNumRuns}
                disabled={isRunning}
              />
              <NumberInput
                label="Warmup Runs"
                value={warmupRuns}
                min={0}
                max={50}
                step={1}
                onChange={setWarmupRuns}
                disabled={isRunning}
              />
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-xs text-slate-400">Image Size</label>
                  <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
                    {imageSize}×{imageSize}
                  </span>
                </div>
                <select
                  value={imageSize}
                  onChange={(e) => setImageSize(Number(e.target.value))}
                  disabled={isRunning}
                  className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                >
                  {[320, 416, 480, 640, 1280].map((s) => (
                    <option key={s} value={s}>
                      {s}×{s}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          <button
            onClick={handleRun}
            disabled={!canRun}
            className={`
              w-full py-2.5 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2
              ${canRun
                ? "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20"
                : "bg-slate-700 text-slate-500 cursor-not-allowed"
              }
            `}
          >
            {isRunning ? (
              <>
                <Spinner />
                Running benchmark…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Run Benchmark
              </>
            )}
          </button>

          {status === "error" && (
            <div className="rounded-lg bg-red-950/50 border border-red-800 px-3 py-2.5 text-xs text-red-300">
              <p className="font-semibold mb-0.5">Error</p>
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* ── Results panel ── */}
        <div className="space-y-4">
          {status === "idle" && (
            <BenchmarkEmptyState />
          )}

          {isRunning && (
            <Card title="Running benchmark…">
              <div className="animate-pulse space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-12 bg-slate-700 rounded-lg" />
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-3 text-center">
                Running {numRuns} timed runs (+{warmupRuns} warmup) per combination…
              </p>
            </Card>
          )}

          {status === "done" && result && (
            <>
              {/* Summary */}
              <Card title={`Results — ${result.num_runs} runs · ${result.image_size}×${result.image_size}px`}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs text-slate-500 border-b border-slate-700">
                        <th className="py-2 text-left font-medium">Model</th>
                        <th className="py-2 text-left font-medium">Backend</th>
                        <th className="py-2 text-right font-medium">Avg (ms)</th>
                        <th className="py-2 text-right font-medium">Min (ms)</th>
                        <th className="py-2 text-right font-medium">Std (ms)</th>
                        <th className="py-2 text-right font-medium">FPS</th>
                        <th className="py-2 text-center font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.results.map((r, i) => (
                        <BenchmarkRow key={i} entry={r} minLatency={minLatency} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Visual comparison */}
              {okResults.length > 0 && (
                <Card title="FPS Comparison">
                  <FpsBarChart entries={okResults} />
                </Card>
              )}

              {/* Latency comparison */}
              {okResults.length > 0 && (
                <Card title="Avg Latency Comparison (ms — lower is better)">
                  <LatencyBarChart entries={okResults} />
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function CheckCard({
  label,
  sublabel,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  sublabel?: string;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      disabled={disabled}
      className={`
        text-left w-full rounded-lg border px-3 py-2.5 transition-all flex items-start gap-2.5
        ${checked
          ? "border-blue-500 bg-blue-500/10 ring-1 ring-blue-500/50"
          : "border-slate-600 bg-slate-800 hover:border-slate-500"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
      `}
    >
      {/* Checkbox */}
      <div
        className={`
          w-4 h-4 rounded border-2 flex-shrink-0 mt-0.5 transition-colors flex items-center justify-center
          ${checked ? "border-blue-500 bg-blue-500" : "border-slate-500"}
        `}
      >
        {checked && (
          <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>
      <div>
        <span className="text-sm font-medium text-slate-100">{label}</span>
        {sublabel && <p className="text-xs text-slate-500 mt-0.5">{sublabel}</p>}
      </div>
    </button>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label className="text-xs text-slate-400">{label}</label>
        <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-slate-600 accent-blue-500 cursor-pointer disabled:opacity-50"
      />
    </div>
  );
}

function BenchmarkRow({ entry, minLatency }: { entry: BenchmarkEntry; minLatency: number }) {
  const isBest = entry.status === "ok" && entry.avg_latency_ms === minLatency;
  return (
    <tr className="border-t border-slate-700/50 hover:bg-slate-800/40 transition-colors">
      <td className="py-2 font-medium text-slate-200">{entry.model_name}</td>
      <td className="py-2 text-slate-400">{entry.backend_type}</td>
      {entry.status === "ok" ? (
        <>
          <td className="py-2 text-right font-mono">
            <span className={isBest ? "text-emerald-400 font-semibold" : "text-slate-300"}>
              {entry.avg_latency_ms.toFixed(1)}
            </span>
            {isBest && (
              <span className="ml-1 text-[10px] bg-emerald-500/20 text-emerald-400 px-1 py-0.5 rounded">
                best
              </span>
            )}
          </td>
          <td className="py-2 text-right font-mono text-slate-400">
            {entry.min_latency_ms.toFixed(1)}
          </td>
          <td className="py-2 text-right font-mono text-slate-500">
            ±{entry.std_latency_ms.toFixed(1)}
          </td>
          <td className="py-2 text-right font-mono text-blue-400">{entry.fps.toFixed(1)}</td>
          <td className="py-2 text-center">
            <span className="text-xs bg-emerald-500/15 text-emerald-400 px-2 py-0.5 rounded-full">
              ok
            </span>
          </td>
        </>
      ) : (
        <>
          <td colSpan={4} className="py-2 text-xs text-slate-500 italic">
            {entry.error ?? "failed"}
          </td>
          <td className="py-2 text-center">
            <span className="text-xs bg-red-500/15 text-red-400 px-2 py-0.5 rounded-full">
              error
            </span>
          </td>
        </>
      )}
    </tr>
  );
}

function FpsBarChart({ entries }: { entries: BenchmarkEntry[] }) {
  const maxFps = Math.max(...entries.map((e) => e.fps), 1);
  return (
    <div className="space-y-2.5">
      {entries
        .slice()
        .sort((a, b) => b.fps - a.fps)
        .map((e, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-36 shrink-0 text-right">
              <span className="text-xs text-slate-400">{e.model_name}</span>
              <span className="text-xs text-slate-600 mx-1">/</span>
              <span className="text-xs text-slate-500">{e.backend_type}</span>
            </div>
            <div className="flex-1 bg-slate-800 rounded-full h-5 overflow-hidden">
              <div
                className="h-full bg-blue-600 rounded-full flex items-center justify-end pr-2 transition-all"
                style={{ width: `${(e.fps / maxFps) * 100}%`, minWidth: "40px" }}
              >
                <span className="text-[10px] font-mono text-white font-semibold">
                  {e.fps.toFixed(1)}
                </span>
              </div>
            </div>
            <div className="w-14 shrink-0 text-xs text-slate-500 text-right">fps</div>
          </div>
        ))}
    </div>
  );
}

function LatencyBarChart({ entries }: { entries: BenchmarkEntry[] }) {
  const maxLat = Math.max(...entries.map((e) => e.avg_latency_ms), 1);
  return (
    <div className="space-y-2.5">
      {entries
        .slice()
        .sort((a, b) => a.avg_latency_ms - b.avg_latency_ms)
        .map((e, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-36 shrink-0 text-right">
              <span className="text-xs text-slate-400">{e.model_name}</span>
              <span className="text-xs text-slate-600 mx-1">/</span>
              <span className="text-xs text-slate-500">{e.backend_type}</span>
            </div>
            <div className="flex-1 bg-slate-800 rounded-full h-5 overflow-hidden">
              <div
                className={`h-full rounded-full flex items-center justify-end pr-2 transition-all ${
                  i === 0 ? "bg-emerald-600" : "bg-slate-600"
                }`}
                style={{ width: `${(e.avg_latency_ms / maxLat) * 100}%`, minWidth: "50px" }}
              >
                <span className="text-[10px] font-mono text-white font-semibold">
                  {e.avg_latency_ms.toFixed(1)}
                </span>
              </div>
            </div>
            <div className="w-14 shrink-0 text-xs text-slate-500 text-right">ms</div>
          </div>
        ))}
    </div>
  );
}

function BenchmarkEmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-72 rounded-xl border-2 border-dashed border-slate-700 bg-slate-800/20 text-center px-6 gap-3">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
        <svg className="w-7 h-7 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <div>
        <p className="text-slate-300 font-medium">No benchmark results yet</p>
        <p className="text-slate-600 text-sm mt-0.5 max-w-xs">
          Select models &amp; backends, configure run count, then click Run Benchmark
        </p>
      </div>
    </div>
  );
}
