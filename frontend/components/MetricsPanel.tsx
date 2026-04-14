"use client";

import type { DetectionResponse, VideoDetectionResponse } from "@/types";

// ── Shared primitives ──────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  unit,
  accent,
}: {
  label: string;
  value: string | number;
  unit?: string;
  accent?: boolean;
}) {
  const display =
    typeof value === "number"
      ? value % 1 === 0
        ? String(value)
        : value < 10
        ? value.toFixed(2)
        : value.toFixed(1)
      : value;

  return (
    <div className="bg-slate-900 rounded-lg px-3 py-2.5 border border-slate-700">
      <p className="text-xs text-slate-500 mb-0.5">{label}</p>
      <p
        className={`text-lg font-semibold font-mono leading-none ${
          accent ? "text-blue-400" : "text-slate-100"
        }`}
      >
        {display}
        {unit && (
          <span className="text-xs font-sans text-slate-500 ml-1">{unit}</span>
        )}
      </p>
    </div>
  );
}

function BadgePill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs rounded-full px-2.5 py-1 bg-slate-700/70 border border-slate-600 text-slate-300">
      <span className="text-slate-500">{label}</span>
      <span className="font-semibold text-slate-200">{value}</span>
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value > 0.7 ? "bg-emerald-500" : value > 0.45 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

// ── Latency breakdown bar chart ────────────────────────────────────────────

function LatencyBreakdown({
  preprocessing,
  inference,
  postprocessing,
}: {
  preprocessing: number;
  inference: number;
  postprocessing: number;
}) {
  const total = preprocessing + inference + postprocessing || 1;
  const segments = [
    { label: "Pre", value: preprocessing, color: "bg-purple-500" },
    { label: "Infer", value: inference, color: "bg-blue-500" },
    { label: "Post", value: postprocessing, color: "bg-teal-500" },
  ];

  return (
    <div>
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        Latency Breakdown
      </p>
      {/* Stacked bar */}
      <div className="flex h-3 rounded-full overflow-hidden gap-px mb-2">
        {segments.map((s) => (
          <div
            key={s.label}
            className={`${s.color} transition-all`}
            style={{ width: `${(s.value / total) * 100}%`, minWidth: s.value > 0 ? "2px" : "0" }}
            title={`${s.label}: ${s.value.toFixed(1)} ms`}
          />
        ))}
      </div>
      {/* Legend */}
      <div className="flex gap-4">
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-sm ${s.color}`} />
            <span className="text-xs text-slate-400">
              {s.label}{" "}
              <span className="font-mono text-slate-300">{s.value.toFixed(1)}ms</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Image metrics ──────────────────────────────────────────────────────────

export function ImageMetricsPanel({ result }: { result: DetectionResponse }) {
  const fps = result.latency_ms > 0 ? 1000 / result.latency_ms : 0;

  return (
    <div className="space-y-4">
      {/* Info pills */}
      <div className="flex flex-wrap gap-2">
        <BadgePill label="model" value={result.model_name} />
        <BadgePill label="backend" value={result.backend_type} />
        <BadgePill label="resolution" value={`${result.image_width}×${result.image_height}`} />
      </div>

      {/* Key metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <MetricCard label="Total Latency" value={result.latency_ms} unit="ms" accent />
        <MetricCard label="Detections" value={result.total_detections} accent />
        <MetricCard label="Throughput" value={fps} unit="FPS" />
        <MetricCard label="Preprocessing" value={result.preprocessing_ms} unit="ms" />
        <MetricCard label="Inference" value={result.inference_ms} unit="ms" />
        <MetricCard label="Postprocessing" value={result.postprocessing_ms} unit="ms" />
      </div>

      {/* Latency breakdown visual */}
      <div className="bg-slate-900 rounded-lg p-3 border border-slate-700">
        <LatencyBreakdown
          preprocessing={result.preprocessing_ms}
          inference={result.inference_ms}
          postprocessing={result.postprocessing_ms}
        />
      </div>

      {/* Detection list */}
      {result.detections.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Detections ({result.detections.length})
          </p>
          <div className="space-y-1 max-h-52 overflow-y-auto pr-1">
            {result.detections
              .slice()
              .sort((a, b) => b.confidence - a.confidence)
              .map((det, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg bg-slate-900 border border-slate-700/80 px-3 py-2 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span className="w-4 h-4 rounded bg-slate-700 text-slate-400 text-center leading-4 text-[10px] font-mono">
                      {i + 1}
                    </span>
                    <span className="font-medium text-slate-200 capitalize">{det.label}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-mono">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                    <ConfidenceBar value={det.confidence} />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Video metrics ──────────────────────────────────────────────────────────

export function VideoMetricsPanel({ result }: { result: VideoDetectionResponse }) {
  return (
    <div className="space-y-4">
      {/* Info pills */}
      <div className="flex flex-wrap gap-2">
        <BadgePill label="model" value={result.model_name} />
        <BadgePill label="backend" value={result.backend_type} />
        <BadgePill label="frames" value={String(result.frame_count)} />
      </div>

      {/* Key metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <MetricCard label="Avg FPS" value={result.average_fps} unit="fps" accent />
        <MetricCard label="Total Detections" value={result.total_detections} accent />
        <MetricCard label="Avg Latency/Frame" value={result.average_latency_per_frame_ms} unit="ms" />
        <MetricCard
          label="Total Duration"
          value={(result.total_latency_ms / 1000).toFixed(2)}
          unit="s"
        />
      </div>

      {/* Per-frame summary table */}
      {result.frames_summary.length > 0 && (
        <div className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
          <div className="px-3 py-2 border-b border-slate-700">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Per-Frame Summary (first 30 frames)
            </p>
          </div>
          <div className="max-h-44 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-900">
                <tr className="text-slate-500 border-b border-slate-700">
                  <th className="py-1.5 px-3 text-left font-medium">Frame</th>
                  <th className="py-1.5 px-3 text-right font-medium">Detections</th>
                  <th className="py-1.5 px-3 text-right font-medium">Latency</th>
                </tr>
              </thead>
              <tbody>
                {result.frames_summary.slice(0, 30).map((f, idx) => (
                  <tr
                    key={f.frame_index}
                    className={`border-t border-slate-800 ${
                      idx % 2 === 0 ? "" : "bg-slate-800/30"
                    }`}
                  >
                    <td className="py-1 px-3 text-slate-500 font-mono">#{f.frame_index}</td>
                    <td className="py-1 px-3 text-right text-slate-300">{f.detections}</td>
                    <td className="py-1 px-3 text-right font-mono text-slate-300">
                      {f.latency_ms.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result.output_path && (
        <div className="rounded-lg bg-slate-900 border border-slate-700 px-4 py-3 flex items-start gap-2">
          <svg className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
              d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
          </svg>
          <div>
            <p className="text-xs text-slate-400">Annotated video saved to:</p>
            <code className="text-xs text-blue-400 break-all">{result.output_path}</code>
          </div>
        </div>
      )}
    </div>
  );
}
