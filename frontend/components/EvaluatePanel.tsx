"use client";

import { useState } from "react";
import { evaluateDataset } from "@/lib/api";
import type { BackendType, EvaluationResult, ModelName } from "@/types";
import { Card, Spinner } from "@/components/ui";

type EvalStatus = "idle" | "running" | "done" | "error";

export default function EvaluatePanel() {
  const [modelName, setModelName] = useState<ModelName>("yolov8");
  const [backendType, setBackendType] = useState<BackendType>("pytorch");
  const [annotationsPath, setAnnotationsPath] = useState(
    "data/annotations/instances_val.json"
  );
  const [imagesDir, setImagesDir] = useState("data/images/val");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [status, setStatus] = useState<EvalStatus>("idle");
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setStatus("running");
    setResult(null);
    setError(null);
    try {
      const r = await evaluateDataset({
        modelName,
        backendType,
        annotationsPath,
        imagesDir,
        confidenceThreshold: confidence,
        iouThreshold: iou,
      });
      setResult(r);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed.");
      setStatus("error");
    }
  };

  const isRunning = status === "running";
  const canRun = annotationsPath.trim() !== "" && imagesDir.trim() !== "" && !isRunning;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-base font-semibold text-slate-200">COCO Evaluation</h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Run inference on your annotated dataset and compute mAP@0.5 and mAP@0.5:0.95
          using pycocotools. Annotations must be in COCO JSON format.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-5">
        {/* ── Config sidebar ── */}
        <div className="space-y-4">
          {/* Dataset paths */}
          <Card title="Dataset">
            <div className="space-y-3">
              <TextInput
                label="Annotations JSON"
                hint="COCO-format annotation file path (relative to backend/)"
                value={annotationsPath}
                onChange={setAnnotationsPath}
                placeholder="data/annotations/instances_val.json"
                disabled={isRunning}
              />
              <TextInput
                label="Images Directory"
                hint="Folder containing evaluation images"
                value={imagesDir}
                onChange={setImagesDir}
                placeholder="data/images/val"
                disabled={isRunning}
              />
            </div>
          </Card>

          {/* Model + backend */}
          <Card title="Model">
            <div className="space-y-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1.5">Model</label>
                <div className="grid grid-cols-2 gap-2">
                  {(["yolov8", "yolov5"] as ModelName[]).map((m) => (
                    <RadioCard
                      key={m}
                      label={m === "yolov8" ? "YOLOv8" : "YOLOv5"}
                      selected={modelName === m}
                      onChange={() => setModelName(m)}
                      disabled={isRunning}
                    />
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1.5">Backend</label>
                <div className="space-y-1.5">
                  {(["pytorch", "torchscript", "onnx"] as BackendType[]).map((b) => (
                    <RadioCard
                      key={b}
                      label={
                        b === "pytorch"
                          ? "PyTorch"
                          : b === "torchscript"
                          ? "TorchScript"
                          : "ONNX Runtime"
                      }
                      selected={backendType === b}
                      onChange={() => setBackendType(b)}
                      disabled={isRunning}
                    />
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Thresholds */}
          <Card title="Thresholds">
            <div className="space-y-3">
              <SliderField
                label="Confidence"
                value={confidence}
                min={0.05}
                max={0.95}
                step={0.05}
                onChange={setConfidence}
                disabled={isRunning}
              />
              <SliderField
                label="IoU (NMS)"
                value={iou}
                min={0.1}
                max={0.9}
                step={0.05}
                onChange={setIou}
                disabled={isRunning}
              />
            </div>
          </Card>

          {/* Run button */}
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
                Evaluating…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                Run Evaluation
              </>
            )}
          </button>

          {status === "error" && error && (
            <div className="rounded-lg bg-red-950/50 border border-red-800 px-3 py-2.5 text-xs text-red-300">
              <p className="font-semibold mb-0.5">Error</p>
              <p className="leading-relaxed">{error}</p>
            </div>
          )}
        </div>

        {/* ── Results panel ── */}
        <div className="space-y-4">
          {status === "idle" && <EvalEmptyState />}

          {isRunning && (
            <Card title="Running evaluation…">
              <div className="animate-pulse space-y-3 py-2">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-12 bg-slate-700 rounded-lg" />
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-3 text-center">
                Running inference on all images in the dataset…
                <br />
                This may take a few minutes depending on dataset size.
              </p>
            </Card>
          )}

          {status === "done" && result && (
            <>
              {/* mAP headline cards */}
              <Card
                title={`Results — ${result.model_name} / ${result.backend_type} · ${result.num_images} images`}
              >
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <MapCard label="mAP @ IoU=0.50" value={result.map_50} color="blue" />
                  <MapCard label="mAP @ IoU=0.50:0.95" value={result.map_50_95} color="purple" />
                </div>

                {/* mAP visual bars */}
                <div className="space-y-3">
                  <MapBar label="mAP@0.5" value={result.map_50} color="bg-blue-500" />
                  <MapBar label="mAP@0.5:0.95" value={result.map_50_95} color="bg-purple-500" />
                </div>
              </Card>

              {/* Latency / FPS */}
              <Card title="Latency & Throughput">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <StatCard
                    label="Avg Latency"
                    value={`${result.average_latency_ms.toFixed(1)} ms`}
                    accent
                  />
                  <StatCard label="FPS" value={result.fps.toFixed(1)} accent />
                  <StatCard label="Images Evaluated" value={String(result.num_images)} />
                  <StatCard
                    label="Total Time"
                    value={`${((result.average_latency_ms * result.num_images) / 1000).toFixed(1)} s`}
                  />
                </div>

                {/* Per-image latency sparkline */}
                {result.per_image_latencies_ms.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs text-slate-500 mb-2">
                      Per-image latency (ms) — {result.per_image_latencies_ms.length} images
                    </p>
                    <LatencySparkline latencies={result.per_image_latencies_ms} />
                  </div>
                )}
              </Card>

              {/* Assignment note */}
              <div className="rounded-xl border border-slate-700 bg-slate-800/30 px-4 py-3">
                <p className="text-xs font-semibold text-slate-400 mb-1.5">
                  Assignment Interpretation
                </p>
                <ul className="text-xs text-slate-500 space-y-1">
                  <li>
                    <span className="text-blue-400 font-mono">mAP@0.5 = {result.map_50.toFixed(4)}</span>
                    {" "}— detection accuracy at IoU threshold 0.50 (PASCAL VOC metric)
                  </li>
                  <li>
                    <span className="text-purple-400 font-mono">mAP@0.5:0.95 = {result.map_50_95.toFixed(4)}</span>
                    {" "}— COCO primary metric, averaged over IoU thresholds 0.50–0.95
                  </li>
                  <li>
                    <span className="text-emerald-400 font-mono">FPS = {result.fps.toFixed(1)}</span>
                    {" "}— inference throughput on this hardware at {result.average_latency_ms.toFixed(1)} ms/image
                  </li>
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function TextInput({
  label,
  hint,
  value,
  onChange,
  placeholder,
  disabled,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  disabled?: boolean;
}) {
  return (
    <div>
      <label className="text-xs font-medium text-slate-300 block mb-1">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-blue-500 disabled:opacity-50 font-mono"
      />
      <p className="text-xs text-slate-600 mt-1">{hint}</p>
    </div>
  );
}

function RadioCard({
  label,
  selected,
  onChange,
  disabled,
}: {
  label: string;
  selected: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      disabled={disabled}
      className={`
        text-left w-full rounded-lg border px-3 py-2 transition-all flex items-center gap-2
        ${selected
          ? "border-blue-500 bg-blue-500/10 ring-1 ring-blue-500/50"
          : "border-slate-600 bg-slate-800 hover:border-slate-500"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
      `}
    >
      <div
        className={`w-3 h-3 rounded-full border-2 flex-shrink-0 transition-colors
          ${selected ? "border-blue-500 bg-blue-500" : "border-slate-500"}`}
      />
      <span className="text-sm text-slate-200">{label}</span>
    </button>
  );
}

function SliderField({
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
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-slate-600 accent-blue-500 cursor-pointer disabled:opacity-50"
      />
    </div>
  );
}

function MapCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "blue" | "purple";
}) {
  const colorClass =
    color === "blue"
      ? "border-blue-700/50 bg-blue-950/30 text-blue-300"
      : "border-purple-700/50 bg-purple-950/30 text-purple-300";
  const valueClass = color === "blue" ? "text-blue-400" : "text-purple-400";

  return (
    <div className={`rounded-xl border p-4 text-center ${colorClass}`}>
      <p className="text-xs mb-2 font-medium">{label}</p>
      <p className={`text-4xl font-bold font-mono ${valueClass}`}>
        {(value * 100).toFixed(1)}
        <span className="text-lg ml-0.5">%</span>
      </p>
      <p className="text-xs text-slate-500 mt-1 font-mono">{value.toFixed(4)}</p>
    </div>
  );
}

function MapBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = Math.min(100, value * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 shrink-0 text-xs text-slate-400 text-right">{label}</div>
      <div className="flex-1 bg-slate-800 rounded-full h-4 overflow-hidden">
        <div
          className={`h-full ${color} rounded-full flex items-center justify-end pr-2 transition-all`}
          style={{ width: `${Math.max(pct, 3)}%` }}
        >
          <span className="text-[10px] font-mono text-white font-semibold">
            {pct.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-slate-900 rounded-lg border border-slate-700 px-3 py-3 text-center">
      <p className="text-xs text-slate-500 mb-0.5">{label}</p>
      <p className={`text-xl font-semibold font-mono ${accent ? "text-blue-400" : "text-slate-100"}`}>
        {value}
      </p>
    </div>
  );
}

function LatencySparkline({ latencies }: { latencies: number[] }) {
  const display = latencies.slice(0, 100);
  const maxLat = Math.max(...display, 1);
  return (
    <div className="bg-slate-900 rounded-lg border border-slate-700 p-3">
      <div className="flex items-end gap-px h-14">
        {display.map((lat, i) => {
          const pct = (lat / maxLat) * 100;
          return (
            <div
              key={i}
              className="flex-1 bg-blue-500/60 rounded-sm"
              style={{ height: `${Math.max(4, pct)}%` }}
              title={`Image ${i + 1}: ${lat.toFixed(1)} ms`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-slate-600 mt-1">
        <span>Image 1</span>
        <span className="font-mono">avg {(latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(1)} ms</span>
        <span>Image {display.length}</span>
      </div>
    </div>
  );
}

function EvalEmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-72 rounded-xl border-2 border-dashed border-slate-700 bg-slate-800/20 text-center px-6 gap-3">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
        <svg className="w-7 h-7 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
      </div>
      <div>
        <p className="text-slate-300 font-medium">No evaluation results yet</p>
        <p className="text-slate-600 text-sm mt-0.5 max-w-xs">
          Point to your COCO annotations and images directory, then click Run Evaluation
        </p>
      </div>
      <div className="text-xs text-slate-600 mt-1 max-w-sm bg-slate-800/50 rounded-lg px-3 py-2 border border-slate-700 text-left">
        <p className="font-mono">data/annotations/instances_val.json</p>
        <p className="font-mono text-slate-700">← COCO format: images + annotations + categories</p>
      </div>
    </div>
  );
}
