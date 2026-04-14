import type {
  BackendType,
  BenchmarkResult,
  DetectionResponse,
  EvaluationResult,
  ModelName,
  VideoDetectionResponse,
} from "@/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // body is not JSON
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export async function getModels(): Promise<
  { model_name: string; backend_type: string; loaded: boolean }[]
> {
  const res = await fetch(`${BASE_URL}/api/models`);
  const data = await handleResponse<{ models: unknown[] }>(res);
  return data.models as { model_name: string; backend_type: string; loaded: boolean }[];
}

// ---------------------------------------------------------------------------
// Image detection
// ---------------------------------------------------------------------------

export async function detectImage(
  file: File,
  options: {
    modelName: ModelName;
    backendType: BackendType;
    confidenceThreshold: number;
    iouThreshold: number;
  }
): Promise<DetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", options.modelName);
  form.append("backend_type", options.backendType);
  form.append("confidence_threshold", String(options.confidenceThreshold));
  form.append("iou_threshold", String(options.iouThreshold));

  const res = await fetch(`${BASE_URL}/api/detect/image`, {
    method: "POST",
    body: form,
  });
  return handleResponse<DetectionResponse>(res);
}

// ---------------------------------------------------------------------------
// Video detection
// ---------------------------------------------------------------------------

export async function detectVideo(
  file: File,
  options: {
    modelName: ModelName;
    backendType: BackendType;
    confidenceThreshold: number;
    iouThreshold: number;
    maxFrames?: number;
  }
): Promise<VideoDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", options.modelName);
  form.append("backend_type", options.backendType);
  form.append("confidence_threshold", String(options.confidenceThreshold));
  form.append("iou_threshold", String(options.iouThreshold));
  form.append("save_output_video", "true");
  if (options.maxFrames !== undefined) {
    form.append("max_frames", String(options.maxFrames));
  }

  const res = await fetch(`${BASE_URL}/api/detect/video`, {
    method: "POST",
    body: form,
  });
  return handleResponse<VideoDetectionResponse>(res);
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

export async function runBenchmark(
  modelNames: ModelName[],
  backendTypes: BackendType[],
  numRuns = 50,
  imageSize = 640,
  warmupRuns = 5
): Promise<BenchmarkResult> {
  const res = await fetch(`${BASE_URL}/api/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_names: modelNames,
      backend_types: backendTypes,
      num_runs: numRuns,
      image_size: imageSize,
      warmup_runs: warmupRuns,
    }),
  });
  return handleResponse<BenchmarkResult>(res);
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export async function evaluateDataset(params: {
  modelName: ModelName;
  backendType: BackendType;
  annotationsPath: string;
  imagesDir: string;
  confidenceThreshold?: number;
  iouThreshold?: number;
}): Promise<EvaluationResult> {
  const res = await fetch(`${BASE_URL}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: params.modelName,
      backend_type: params.backendType,
      annotations_path: params.annotationsPath,
      images_dir: params.imagesDir,
      confidence_threshold: params.confidenceThreshold ?? 0.25,
      iou_threshold: params.iouThreshold ?? 0.45,
    }),
  });
  return handleResponse<EvaluationResult>(res);
}
