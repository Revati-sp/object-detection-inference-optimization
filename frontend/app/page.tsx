"use client";

import { useState, useCallback } from "react";
import ModelSelector from "@/components/ModelSelector";
import UploadForm from "@/components/UploadForm";
import ImageResultViewer from "@/components/ImageResultViewer";
import VideoResultViewer from "@/components/VideoResultViewer";
import { ImageMetricsPanel, VideoMetricsPanel } from "@/components/MetricsPanel";
import BenchmarkPanel from "@/components/BenchmarkPanel";
import EvaluatePanel from "@/components/EvaluatePanel";
import { Card, Spinner } from "@/components/ui";
import { detectImage, detectVideo } from "@/lib/api";
import type {
  DetectionResponse,
  InferenceSettings,
  MediaMode,
  VideoDetectionResponse,
} from "@/types";

// ── Tab setup ──────────────────────────────────────────────────────────────

type Tab = "detect" | "benchmark" | "evaluate";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  {
    id: "detect",
    label: "Detect",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
          d="M15 10l4.553-2.069A1 1 0 0121 8.87V15.13a1 1 0 01-1.447.9L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
      </svg>
    ),
  },
  {
    id: "benchmark",
    label: "Benchmark",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: "evaluate",
    label: "Evaluate",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
  },
];

// ── Detect tab state ───────────────────────────────────────────────────────

const DEFAULT_SETTINGS: InferenceSettings = {
  modelName: "yolov8",
  backendType: "pytorch",
  confidenceThreshold: 0.25,
  iouThreshold: 0.45,
};

type Status = "idle" | "loading" | "success" | "error";

// ── Root page ──────────────────────────────────────────────────────────────

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<Tab>("detect");

  return (
    <div className="space-y-5">
      {/* Page heading */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Object Detection</h1>
        <p className="text-slate-400 text-sm mt-1">
          YOLOv8 &amp; YOLOv5 · PyTorch / TorchScript / ONNX Runtime
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-slate-700">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg
              border border-transparent border-b-0 -mb-px transition-colors
              ${activeTab === tab.id
                ? "bg-slate-800 border-slate-700 text-slate-100"
                : "text-slate-400 hover:text-slate-300 hover:bg-slate-800/50"
              }
            `}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "detect" && <DetectTab />}
      {activeTab === "benchmark" && <BenchmarkPanel />}
      {activeTab === "evaluate" && <EvaluatePanel />}
    </div>
  );
}

// ── Detect Tab ─────────────────────────────────────────────────────────────

function DetectTab() {
  const [mode, setMode] = useState<MediaMode>("image");
  const [settings, setSettings] = useState<InferenceSettings>(DEFAULT_SETTINGS);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<DetectionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoDetectionResponse | null>(null);

  const handleModeChange = (newMode: MediaMode) => {
    setMode(newMode);
    resetResults();
  };

  const resetResults = () => {
    setSelectedFile(null);
    setImagePreviewUrl(null);
    setImageResult(null);
    setVideoResult(null);
    setError(null);
    setStatus("idle");
  };

  const handleFileSelect = useCallback(
    (file: File) => {
      setSelectedFile(file);
      setImageResult(null);
      setVideoResult(null);
      setError(null);
      setStatus("idle");

      if (mode === "image") {
        // Revoke any existing preview URL to avoid memory leaks
        setImagePreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(file);
        });
      } else {
        setImagePreviewUrl(null);
      }
    },
    [mode]
  );

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setStatus("loading");
    setError(null);
    setImageResult(null);
    setVideoResult(null);

    try {
      if (mode === "image") {
        const result = await detectImage(selectedFile, {
          modelName: settings.modelName,
          backendType: settings.backendType,
          confidenceThreshold: settings.confidenceThreshold,
          iouThreshold: settings.iouThreshold,
        });
        setImageResult(result);
      } else {
        const result = await detectVideo(selectedFile, {
          modelName: settings.modelName,
          backendType: settings.backendType,
          confidenceThreshold: settings.confidenceThreshold,
          iouThreshold: settings.iouThreshold,
        });
        setVideoResult(result);
      }
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed.");
      setStatus("error");
    }
  };

  const isLoading = status === "loading";
  const canRun = !!selectedFile && !isLoading;
  const hasResult = !!imageResult || !!videoResult;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
      {/* ── Sidebar ── */}
      <aside className="space-y-4">
        <Card title="Input">
          <UploadForm
            mode={mode}
            onModeChange={(m) => handleModeChange(m)}
            onFileSelect={handleFileSelect}
            disabled={isLoading}
          />
        </Card>

        <Card title="Configuration">
          <ModelSelector
            settings={settings}
            onChange={(u) => setSettings((p) => ({ ...p, ...u }))}
            disabled={isLoading}
          />
        </Card>

        {/* Action buttons */}
        <div className="space-y-2">
          <button
            onClick={handleRunInference}
            disabled={!canRun}
            className={`
              w-full py-2.5 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2
              ${canRun
                ? "bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white shadow-lg shadow-blue-500/20"
                : "bg-slate-700 text-slate-500 cursor-not-allowed"
              }
            `}
          >
            {isLoading ? (
              <>
                <Spinner />
                Running inference…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Run Inference
              </>
            )}
          </button>

          {hasResult && (
            <button
              onClick={resetResults}
              className="w-full py-2 rounded-lg text-sm text-slate-400 hover:text-slate-200 border border-slate-700 hover:border-slate-500 transition-colors"
            >
              Clear results
            </button>
          )}
        </div>

        {/* Error message */}
        {status === "error" && error && (
          <div className="rounded-lg bg-red-950/50 border border-red-800 px-4 py-3 text-sm text-red-300">
            <p className="font-semibold mb-0.5 flex items-center gap-1.5">
              <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Error
            </p>
            <p className="text-xs leading-relaxed text-red-400">{error}</p>
          </div>
        )}
      </aside>

      {/* ── Main panel ── */}
      <div className="space-y-4 min-h-[420px]">
        {/* Empty state */}
        {status === "idle" && !selectedFile && <EmptyState mode={mode} />}

        {/* Image preview before inference */}
        {mode === "image" && imagePreviewUrl && !imageResult && !isLoading && (
          <Card title="Preview">
            <img
              src={imagePreviewUrl}
              alt="Preview"
              className="w-full rounded-lg max-h-[480px] object-contain bg-black/20"
            />
          </Card>
        )}

        {/* Video selected — waiting for inference */}
        {mode === "video" && selectedFile && !videoResult && !isLoading && (
          <Card title="Video selected">
            <div className="flex items-center gap-3 py-2">
              <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center shrink-0">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
                    d="M15 10l4.553-2.069A1 1 0 0121 8.87V15.13a1 1 0 01-1.447.9L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">{selectedFile.name}</p>
                <p className="text-xs text-slate-500">{formatBytes(selectedFile.size)}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Loading skeleton */}
        {isLoading && (
          <Card title={`Running ${settings.modelName} / ${settings.backendType}…`}>
            <div className="animate-pulse space-y-3">
              <div className="h-56 bg-slate-700 rounded-lg" />
              <div className="grid grid-cols-3 gap-2">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-14 bg-slate-700 rounded-lg" />
                ))}
              </div>
            </div>
          </Card>
        )}

        {/* Image results */}
        {imageResult && imagePreviewUrl && (
          <>
            <Card
              title={`Detection Result — ${imageResult.total_detections} object${imageResult.total_detections !== 1 ? "s" : ""} found`}
              action={
                <DownloadCanvasButton
                  imageUrl={imagePreviewUrl}
                  result={imageResult}
                  filename={`detection_${settings.modelName}_${settings.backendType}.jpg`}
                />
              }
            >
              <ImageResultViewer imageUrl={imagePreviewUrl} result={imageResult} />
            </Card>

            <Card title="Performance Metrics">
              <ImageMetricsPanel result={imageResult} />
            </Card>
          </>
        )}

        {/* Video results */}
        {videoResult && (
          <>
            <Card title="Video Analysis">
              <VideoResultViewer result={videoResult} />
            </Card>
            <Card title="Performance Metrics">
              <VideoMetricsPanel result={videoResult} />
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

// ── Download button — triggers canvas-to-blob save ─────────────────────────

function DownloadCanvasButton({
  imageUrl,
  result,
  filename,
}: {
  imageUrl: string;
  result: import("@/types").DetectionResponse;
  filename: string;
}) {
  const handleDownload = () => {
    // Redraw onto a temp canvas at full resolution, then download
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = result.image_width;
      canvas.height = result.image_height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);

      const PALETTE = [
        "#FF3838","#FF9D97","#FF701F","#FFB21D","#CFD231","#48F90A","#92CC17",
        "#3DDB86","#1A9334","#00D4BB","#2C99A8","#00C2FF","#344593","#6473FF",
        "#0018EC","#8438FF","#520085","#CB38FF","#FF95C8","#FF37C7",
      ];

      const lw = Math.max(2, Math.round(result.image_width / 400));
      const fs = Math.max(12, Math.round(result.image_width / 50));
      ctx.font = `bold ${fs}px ui-monospace, monospace`;

      for (const det of result.detections) {
        const { x1, y1, x2, y2 } = det.bbox;
        const color = PALETTE[det.class_id % PALETTE.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
        const tw = ctx.measureText(label).width + 10;
        const th = fs + 6;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - th, tw, th);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + 4, y1 - 4);
      }

      canvas.toBlob((blob) => {
        if (!blob) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
      }, "image/jpeg", 0.92);
    };
    img.src = imageUrl;
  };

  return (
    <button
      onClick={handleDownload}
      className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-blue-400 transition-colors"
    >
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Download
    </button>
  );
}

function EmptyState({ mode }: { mode: MediaMode }) {
  return (
    <div className="flex flex-col items-center justify-center h-72 rounded-xl border-2 border-dashed border-slate-700 bg-slate-800/20 text-center px-6 gap-3">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
        {mode === "image" ? (
          <svg className="w-7 h-7 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        ) : (
          <svg className="w-7 h-7 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15 10l4.553-2.069A1 1 0 0121 8.87V15.13a1 1 0 01-1.447.9L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
          </svg>
        )}
      </div>
      <div>
        <p className="text-slate-300 font-medium">
          Upload {mode === "image" ? "an image" : "a video"} to get started
        </p>
        <p className="text-slate-600 text-sm mt-0.5">
          Choose a model &amp; backend, then click Run Inference
        </p>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
