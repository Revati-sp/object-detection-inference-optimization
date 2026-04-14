// ── API enum mirrors ────────────────────────────────────────────────────────

export type ModelName = "yolov8" | "yolov5";
export type BackendType = "pytorch" | "torchscript" | "onnx";

// ── Detection primitives ────────────────────────────────────────────────────

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface Detection {
  bbox: BoundingBox;
  label: string;
  class_id: number;
  confidence: number;
}

// ── Image detection response ────────────────────────────────────────────────

export interface DetectionResponse {
  model_name: string;
  backend_type: string;
  image_width: number;
  image_height: number;
  detections: Detection[];
  total_detections: number;
  latency_ms: number;
  preprocessing_ms: number;
  inference_ms: number;
  postprocessing_ms: number;
}

// ── Video detection response ────────────────────────────────────────────────

export interface FrameSummary {
  frame_index: number;
  detections: number;
  latency_ms: number;
}

export interface VideoDetectionResponse {
  model_name: string;
  backend_type: string;
  frame_count: number;
  average_fps: number;
  total_latency_ms: number;
  average_latency_per_frame_ms: number;
  total_detections: number;
  output_path: string | null;
  frames_summary: FrameSummary[];
}

// ── Benchmark ───────────────────────────────────────────────────────────────

export interface BenchmarkEntry {
  model_name: string;
  backend_type: string;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  std_latency_ms: number;
  fps: number;
  image_size: number;
  num_runs: number;
  status: "ok" | "error";
  error?: string;
}

export interface BenchmarkResult {
  results: BenchmarkEntry[];
  image_size: number;
  num_runs: number;
  warmup_runs: number;
}

// ── Evaluation ──────────────────────────────────────────────────────────────

export interface EvaluationResult {
  model_name: string;
  backend_type: string;
  num_images: number;
  map_50: number;
  map_50_95: number;
  per_image_latencies_ms: number[];
  average_latency_ms: number;
  fps: number;
}

// ── UI state ─────────────────────────────────────────────────────────────────

export type MediaMode = "image" | "video";

export interface InferenceSettings {
  modelName: ModelName;
  backendType: BackendType;
  confidenceThreshold: number;
  iouThreshold: number;
}
