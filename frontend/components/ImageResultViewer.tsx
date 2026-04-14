"use client";

import { useEffect, useRef, useState } from "react";
import type { Detection, DetectionResponse } from "@/types";

interface Props {
  imageUrl: string;
  result: DetectionResponse;
}

const PALETTE = [
  "#FF3838","#FF9D97","#FF701F","#FFB21D","#CFD231","#48F90A","#92CC17",
  "#3DDB86","#1A9334","#00D4BB","#2C99A8","#00C2FF","#344593","#6473FF",
  "#0018EC","#8438FF","#520085","#CB38FF","#FF95C8","#FF37C7",
];

function getColor(classId: number): string {
  return PALETTE[classId % PALETTE.length];
}

function drawBoxes(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  detections: Detection[],
  imgWidth: number,
  imgHeight: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const scaleX = canvas.width / imgWidth;
  const scaleY = canvas.height / imgHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (detections.length === 0) return;

  const lineW = Math.max(1.5, canvas.width / 500);
  const fontSize = Math.max(10, Math.round(canvas.width / 65));
  ctx.font = `bold ${fontSize}px ui-monospace, monospace`;

  for (const det of detections) {
    const { x1, y1, x2, y2 } = det.bbox;
    const sx = x1 * scaleX;
    const sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    const color = getColor(det.class_id);

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.strokeRect(sx, sy, sw, sh);

    // Semi-transparent fill for the box
    ctx.fillStyle = color + "1A"; // ~10% opacity
    ctx.fillRect(sx, sy, sw, sh);

    // Label pill
    const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
    const textW = ctx.measureText(label).width;
    const pillH = fontSize + 8;
    const pillW = textW + 12;

    // Position label above the box, clamped so it doesn't go off canvas
    const labelY = sy > pillH ? sy - pillH : sy + sh;

    ctx.fillStyle = color;
    // Rounded pill
    roundRect(ctx, sx, labelY, pillW, pillH, 3);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, sx + 6, labelY + pillH - 5);
  }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

export default function ImageResultViewer({ imageUrl, result }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setLoaded(false);
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const img = new Image();
    img.onload = () => {
      const maxW = 800;
      const scale = Math.min(1, maxW / img.naturalWidth);
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);
      drawBoxes(canvas, img, result.detections, result.image_width, result.image_height);
      setLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl, result]);

  return (
    <div className="relative rounded-lg overflow-hidden bg-black/30 border border-slate-700">
      <canvas
        ref={canvasRef}
        className="block w-full"
        style={{ imageRendering: "auto" }}
      />

      {/* "No detections" overlay — correctly uses relative parent */}
      {loaded && result.detections.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="text-center">
            <p className="text-slate-300 font-medium">No detections found</p>
            <p className="text-slate-500 text-sm mt-0.5">
              Try lowering the confidence threshold
            </p>
          </div>
        </div>
      )}

      {/* Detection count badge */}
      {loaded && result.detections.length > 0 && (
        <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-sm rounded-md px-2 py-1 text-xs text-slate-200 font-medium">
          {result.detections.length} detection{result.detections.length !== 1 ? "s" : ""}
        </div>
      )}
    </div>
  );
}
