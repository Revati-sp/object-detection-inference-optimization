"use client";

import type { BackendType, InferenceSettings, ModelName } from "@/types";

interface Props {
  settings: InferenceSettings;
  onChange: (updated: Partial<InferenceSettings>) => void;
  disabled?: boolean;
}

const MODELS: { value: ModelName; label: string; description: string }[] = [
  { value: "yolov8", label: "YOLOv8", description: "Ultralytics · state-of-the-art" },
  { value: "yolov5", label: "YOLOv5", description: "torch.hub · battle-tested" },
];

const BACKENDS: { value: BackendType; label: string; description: string }[] = [
  { value: "pytorch", label: "PyTorch", description: "Baseline — Ultralytics / torch.hub" },
  { value: "torchscript", label: "TorchScript", description: "Compiled — requires export step" },
  { value: "onnx", label: "ONNX Runtime", description: "Optimized — CPU or CUDA provider" },
];

function RadioCard({
  value,
  selected,
  label,
  description,
  onChange,
  disabled,
}: {
  value: string;
  selected: boolean;
  label: string;
  description: string;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      disabled={disabled}
      className={`
        text-left w-full rounded-lg border px-3 py-2.5 transition-all
        ${selected
          ? "border-blue-500 bg-blue-500/10 ring-1 ring-blue-500"
          : "border-slate-600 bg-slate-800 hover:border-slate-500"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
      `}
    >
      <div className="flex items-center gap-2">
        <div
          className={`w-3.5 h-3.5 rounded-full border-2 flex-shrink-0 transition-colors
            ${selected ? "border-blue-500 bg-blue-500" : "border-slate-500"}`}
        />
        <span className="text-sm font-medium text-slate-100">{label}</span>
      </div>
      <p className="text-xs text-slate-400 mt-0.5 ml-5">{description}</p>
    </button>
  );
}

function SliderInput({
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
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-slate-300">{label}</label>
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

export default function ModelSelector({ settings, onChange, disabled }: Props) {
  return (
    <div className="space-y-5">
      {/* Model */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Model
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {MODELS.map((m) => (
            <RadioCard
              key={m.value}
              value={m.value}
              selected={settings.modelName === m.value}
              label={m.label}
              description={m.description}
              onChange={() => onChange({ modelName: m.value })}
              disabled={disabled}
            />
          ))}
        </div>
      </div>

      {/* Backend */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Inference Backend
        </h3>
        <div className="space-y-2">
          {BACKENDS.map((b) => (
            <RadioCard
              key={b.value}
              value={b.value}
              selected={settings.backendType === b.value}
              label={b.label}
              description={b.description}
              onChange={() => onChange({ backendType: b.value })}
              disabled={disabled}
            />
          ))}
        </div>
      </div>

      {/* Thresholds */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Thresholds
        </h3>
        <SliderInput
          label="Confidence"
          value={settings.confidenceThreshold}
          min={0.05}
          max={0.95}
          step={0.05}
          onChange={(v) => onChange({ confidenceThreshold: v })}
          disabled={disabled}
        />
        <SliderInput
          label="IoU (NMS)"
          value={settings.iouThreshold}
          min={0.1}
          max={0.9}
          step={0.05}
          onChange={(v) => onChange({ iouThreshold: v })}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
