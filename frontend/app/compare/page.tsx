"use client";

import { useRef, useState } from "react";
import type { BackendType, CompareEntry, CompareResponse, ModelName } from "@/types";
import { compareModels } from "@/lib/api";
import LatencyBadge from "@/components/LatencyBadge";
import ImageResultViewer from "@/components/ImageResultViewer";
import type { DetectionResponse } from "@/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Combo {
  id: number;
  modelName: ModelName;
  backendType: BackendType;
}

const MODEL_OPTIONS: ModelName[] = ["yolov8", "yolov5", "rtdetr"];
const BACKEND_OPTIONS: BackendType[] = ["pytorch", "torchscript", "onnx", "onnx_quant", "coreml", "tensorrt"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function entryToDetectionResponse(entry: CompareEntry, w: number, h: number): DetectionResponse {
  return {
    model_name: entry.model_name,
    backend_type: entry.backend_type,
    image_width: w,
    image_height: h,
    detections: entry.detections,
    total_detections: entry.total_detections,
    latency_ms: entry.latency_ms,
    preprocessing_ms: entry.preprocessing_ms,
    inference_ms: entry.inference_ms,
    postprocessing_ms: entry.postprocessing_ms,
  };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ComboRow({
  combo,
  index,
  total,
  onChange,
  onRemove,
}: {
  combo: Combo;
  index: number;
  total: number;
  onChange: (id: number, field: "modelName" | "backendType", value: string) => void;
  onRemove: (id: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-4 text-right">{index + 1}</span>
      <select
        value={combo.modelName}
        onChange={(e) => onChange(combo.id, "modelName", e.target.value)}
        className="flex-1 bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
      >
        {MODEL_OPTIONS.map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
      <select
        value={combo.backendType}
        onChange={(e) => onChange(combo.id, "backendType", e.target.value)}
        className="flex-1 bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
      >
        {BACKEND_OPTIONS.map((b) => (
          <option key={b} value={b}>{b}</option>
        ))}
      </select>
      {total > 1 && (
        <button
          type="button"
          onClick={() => onRemove(combo.id)}
          className="text-slate-500 hover:text-red-400 transition-colors text-lg leading-none px-1"
          aria-label="Remove"
        >
          ×
        </button>
      )}
    </div>
  );
}

function ResultCard({
  entry,
  imageUrl,
  imageWidth,
  imageHeight,
}: {
  entry: CompareEntry;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
}) {
  const fps = entry.latency_ms > 0 ? 1000 / entry.latency_ms : 0;
  const detResponse = entryToDetectionResponse(entry, imageWidth, imageHeight);

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/60 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between gap-3">
        <div>
          <span className="font-semibold text-slate-100 text-sm">{entry.model_name}</span>
          <span className="text-slate-500 text-xs ml-2">/ {entry.backend_type}</span>
        </div>
        {entry.status === "ok" ? (
          <LatencyBadge latencyMs={entry.latency_ms} fps={fps} size="sm" />
        ) : (
          <span className="text-xs text-red-400 bg-red-400/10 border border-red-500/30 rounded px-2 py-1">
            error
          </span>
        )}
      </div>

      {entry.status === "ok" ? (
        <>
          <div className="p-3">
            <ImageResultViewer imageUrl={imageUrl} result={detResponse} />
          </div>
          <div className="px-4 pb-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
            <div>
              <div className="text-slate-500">Detections</div>
              <div className="text-slate-200 font-mono">{entry.total_detections}</div>
            </div>
            <div>
              <div className="text-slate-500">Inference</div>
              <div className="text-slate-200 font-mono">{entry.inference_ms.toFixed(1)} ms</div>
            </div>
            <div>
              <div className="text-slate-500">Pre+Post</div>
              <div className="text-slate-200 font-mono">
                {(entry.preprocessing_ms + entry.postprocessing_ms).toFixed(1)} ms
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="p-4 text-sm text-red-400">
          <p className="font-medium mb-1">Failed to run inference</p>
          <p className="text-xs text-red-400/70 font-mono break-all">{entry.error}</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

let _nextId = 3;

export default function ComparePage() {
  const [combos, setCombos] = useState<Combo[]>([
    { id: 1, modelName: "yolov8", backendType: "pytorch" },
    { id: 2, modelName: "yolov5", backendType: "pytorch" },
  ]);
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function addCombo() {
    setCombos((prev) => [...prev, { id: _nextId++, modelName: "yolov8", backendType: "onnx" }]);
  }

  function removeCombo(id: number) {
    setCombos((prev) => prev.filter((c) => c.id !== id));
  }

  function updateCombo(id: number, field: "modelName" | "backendType", value: string) {
    setCombos((prev) =>
      prev.map((c) => (c.id === id ? { ...c, [field]: value } : c))
    );
  }

  async function handleRun() {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await compareModels(
        imageFile,
        combos.map((c) => ({ modelName: c.modelName, backendType: c.backendType })),
        { confidenceThreshold: confidence, iouThreshold: iou }
      );
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Model Comparison</h1>
        <p className="text-slate-400 text-sm mt-1">
          Run multiple model/backend combos on the same image and compare latency and detections side-by-side.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: config panel */}
        <div className="space-y-5">
          {/* Image upload */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 space-y-3">
            <h2 className="text-sm font-semibold text-slate-300">Image</h2>
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileRef.current?.click()}
              className="border-2 border-dashed border-slate-600 hover:border-blue-500 rounded-lg p-6 text-center cursor-pointer transition-colors"
            >
              {imageUrl ? (
                <img src={imageUrl} alt="preview" className="max-h-32 mx-auto rounded object-contain" />
              ) : (
                <div className="text-slate-500 text-sm">Drop an image or click to browse</div>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
          </div>

          {/* Combos */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-300">Combos</h2>
              <div className="grid grid-cols-2 gap-1 text-xs text-slate-500 mr-7 w-[calc(100%-3rem)]">
                <span className="ml-4">Model</span>
                <span>Backend</span>
              </div>
            </div>
            <div className="space-y-2">
              {combos.map((c, i) => (
                <ComboRow
                  key={c.id}
                  combo={c}
                  index={i}
                  total={combos.length}
                  onChange={updateCombo}
                  onRemove={removeCombo}
                />
              ))}
            </div>
            {combos.length < 6 && (
              <button
                type="button"
                onClick={addCombo}
                className="w-full text-xs text-blue-400 hover:text-blue-300 border border-dashed border-slate-600 hover:border-blue-500 rounded-lg py-2 transition-colors"
              >
                + Add combo
              </button>
            )}
          </div>

          {/* Thresholds */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 space-y-3">
            <h2 className="text-sm font-semibold text-slate-300">Thresholds</h2>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Confidence</span>
                <span className="font-mono text-blue-400">{confidence.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.05} max={0.95} step={0.05}
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none bg-slate-600 accent-blue-500"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">IoU (NMS)</span>
                <span className="font-mono text-blue-400">{iou.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0.1} max={0.9} step={0.05}
                value={iou}
                onChange={(e) => setIou(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none bg-slate-600 accent-blue-500"
              />
            </div>
          </div>

          {/* Run button */}
          <button
            type="button"
            onClick={handleRun}
            disabled={!imageFile || loading}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm transition-colors"
          >
            {loading ? "Running…" : "Run Comparison"}
          </button>

          {error && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Right: results */}
        <div className="lg:col-span-2">
          {!result && !loading && (
            <div className="h-full flex items-center justify-center text-slate-600 text-sm border border-dashed border-slate-700 rounded-xl p-12">
              Results will appear here after running a comparison.
            </div>
          )}

          {loading && (
            <div className="h-full flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-700 rounded-xl p-12">
              <div className="text-center space-y-2">
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
                <p>Running {combos.length} combo{combos.length !== 1 ? "s" : ""}…</p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Summary bar */}
              <div className="flex items-center gap-4 flex-wrap text-xs text-slate-400">
                <span>
                  {result.comparisons.filter((c) => c.status === "ok").length} / {result.comparisons.length} succeeded
                </span>
                {result.comparisons.filter((c) => c.status === "ok").length > 0 && (() => {
                  const ok = result.comparisons.filter((c) => c.status === "ok");
                  const fastest = ok.reduce((a, b) => a.latency_ms < b.latency_ms ? a : b);
                  return (
                    <span className="text-green-400">
                      Fastest: {fastest.model_name}/{fastest.backend_type} — {fastest.latency_ms.toFixed(1)} ms
                    </span>
                  );
                })()}
              </div>

              {/* Grid of result cards */}
              <div className={`grid gap-4 ${result.comparisons.length === 1 ? "grid-cols-1" : "grid-cols-1 xl:grid-cols-2"}`}>
                {result.comparisons.map((entry, i) => (
                  <ResultCard
                    key={i}
                    entry={entry}
                    imageUrl={imageUrl!}
                    imageWidth={result.image_width}
                    imageHeight={result.image_height}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
