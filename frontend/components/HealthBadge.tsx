"use client";

import { useEffect, useState } from "react";

type Health = "checking" | "ok" | "down";

export default function HealthBadge() {
  const [health, setHealth] = useState<Health>("checking");

  useEffect(() => {
    const check = () => {
      fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/health`)
        .then((r) => (r.ok ? setHealth("ok") : setHealth("down")))
        .catch(() => setHealth("down"));
    };
    check();
    // Re-check every 30 s
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  if (health === "checking") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-slate-500">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-pulse" />
        API…
      </span>
    );
  }

  if (health === "down") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-red-400">
        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
        API offline
      </span>
    );
  }

  return (
    <span className="flex items-center gap-1.5 text-xs text-emerald-400">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
      API online
    </span>
  );
}
