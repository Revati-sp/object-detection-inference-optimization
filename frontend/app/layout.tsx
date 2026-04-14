import type { Metadata } from "next";
import "./globals.css";
import HealthBadge from "@/components/HealthBadge";

export const metadata: Metadata = {
  title: "Object Detection — Inference Optimization",
  description:
    "YOLOv8 & YOLOv5 inference via PyTorch, TorchScript, and ONNX Runtime",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100 antialiased">
        <header className="border-b border-slate-700 bg-slate-800/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-3">
            {/* Logo mark */}
            <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center text-white font-bold text-sm select-none shrink-0">
              OD
            </div>

            <span className="font-semibold text-slate-100 tracking-tight">
              Object Detection
            </span>
            <span className="text-slate-600 hidden sm:inline">·</span>
            <span className="text-slate-500 text-sm hidden sm:inline">
              Inference Optimization
            </span>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Backend health */}
            <HealthBadge />

            {/* API docs link */}
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1 text-xs text-slate-400 hover:text-blue-400 transition-colors border border-slate-600 rounded px-2 py-1"
            >
              API Docs
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
