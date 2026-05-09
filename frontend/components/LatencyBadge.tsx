"use client";

interface Props {
  latencyMs: number;
  fps?: number;
  size?: "sm" | "md";
}

function getColor(ms: number): string {
  if (ms <= 20) return "text-green-400 bg-green-400/10 border-green-500/30";
  if (ms <= 60) return "text-yellow-400 bg-yellow-400/10 border-yellow-500/30";
  return "text-red-400 bg-red-400/10 border-red-500/30";
}

export default function LatencyBadge({ latencyMs, fps, size = "md" }: Props) {
  const color = getColor(latencyMs);
  const textSize = size === "sm" ? "text-xs" : "text-sm";

  return (
    <div className={`inline-flex items-center gap-2 rounded-md border px-2 py-1 font-mono ${color}`}>
      <span className={`font-semibold ${textSize}`}>{latencyMs.toFixed(1)} ms</span>
      {fps !== undefined && (
        <span className={`text-xs opacity-70`}>{fps.toFixed(1)} FPS</span>
      )}
    </div>
  );
}
