"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";
import type { MediaMode } from "@/types";

interface Props {
  mode: MediaMode;
  onModeChange: (mode: MediaMode) => void;
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED: Record<MediaMode, string> = {
  image: "image/jpeg,image/png,image/bmp,image/webp",
  video: "video/mp4,video/avi,video/quicktime,video/webm,video/x-matroska",
};

export default function UploadForm({ mode, onModeChange, onFileSelect, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setFileName(file.name);
    onFileSelect(file);
  };

  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex rounded-lg overflow-hidden border border-slate-600 w-fit">
        {(["image", "video"] as MediaMode[]).map((m) => (
          <button
            key={m}
            type="button"
            disabled={disabled}
            onClick={() => {
              onModeChange(m);
              setFileName(null);
            }}
            className={`
              px-5 py-1.5 text-sm font-medium transition-colors capitalize
              ${mode === m
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`
          relative flex flex-col items-center justify-center gap-2
          rounded-xl border-2 border-dashed p-8 cursor-pointer
          transition-all select-none
          ${dragging ? "border-blue-500 bg-blue-500/10" : "border-slate-600 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800"}
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED[mode]}
          className="hidden"
          onChange={onInputChange}
          disabled={disabled}
        />
        <UploadIcon />
        {fileName ? (
          <div className="text-center">
            <p className="text-sm font-medium text-slate-200 truncate max-w-xs">{fileName}</p>
            <p className="text-xs text-slate-500 mt-0.5">Click or drop to replace</p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-sm font-medium text-slate-300">
              Drop {mode === "image" ? "an image" : "a video"} here
            </p>
            <p className="text-xs text-slate-500 mt-0.5">
              {mode === "image" ? "JPEG, PNG, BMP, WebP" : "MP4, AVI, MOV, WebM, MKV"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="w-8 h-8 text-slate-500"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
      />
    </svg>
  );
}
