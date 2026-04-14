"use client";

import type { VideoDetectionResponse } from "@/types";

interface Props {
  result: VideoDetectionResponse;
}

export default function VideoResultViewer({ result }: Props) {
  // Sparkline-style bar chart of detections per frame
  const maxDets = Math.max(...result.frames_summary.map((f) => f.detections), 1);
  const displayFrames = result.frames_summary.slice(0, 80);

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Frames" value={result.frame_count.toString()} />
        <StatCard
          label="Avg FPS"
          value={result.average_fps.toFixed(1)}
          accent
        />
        <StatCard
          label="Avg Latency"
          value={`${result.average_latency_per_frame_ms.toFixed(1)} ms`}
        />
        <StatCard
          label="Total Detections"
          value={result.total_detections.toString()}
        />
      </div>

      {/* Per-frame detection sparkline */}
      {displayFrames.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Detections per frame
          </h4>
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
            <div className="flex items-end gap-px h-16">
              {displayFrames.map((f) => {
                const pct = maxDets > 0 ? (f.detections / maxDets) * 100 : 0;
                return (
                  <div
                    key={f.frame_index}
                    className="flex-1 bg-blue-500/70 rounded-sm transition-all"
                    style={{ height: `${Math.max(4, pct)}%` }}
                    title={`Frame ${f.frame_index}: ${f.detections} detections, ${f.latency_ms.toFixed(1)} ms`}
                  />
                );
              })}
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>Frame 0</span>
              <span>Frame {displayFrames[displayFrames.length - 1]?.frame_index}</span>
            </div>
          </div>
        </div>
      )}

      {/* Latency sparkline */}
      {displayFrames.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Latency per frame (ms)
          </h4>
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
            <div className="flex items-end gap-px h-16">
              {displayFrames.map((f) => {
                const maxLat = Math.max(...displayFrames.map((x) => x.latency_ms));
                const pct = maxLat > 0 ? (f.latency_ms / maxLat) * 100 : 0;
                return (
                  <div
                    key={f.frame_index}
                    className="flex-1 bg-emerald-500/60 rounded-sm"
                    style={{ height: `${Math.max(4, pct)}%` }}
                    title={`Frame ${f.frame_index}: ${f.latency_ms.toFixed(1)} ms`}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {result.output_path && (
        <div className="rounded-lg bg-slate-800 border border-slate-700 px-4 py-3">
          <p className="text-xs text-slate-400 mb-1">Annotated video saved at:</p>
          <code className="text-sm text-blue-400 break-all">{result.output_path}</code>
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 px-3 py-3 text-center">
      <p className="text-xs text-slate-400 mb-0.5">{label}</p>
      <p className={`text-xl font-semibold font-mono ${accent ? "text-blue-400" : "text-slate-100"}`}>
        {value}
      </p>
    </div>
  );
}
