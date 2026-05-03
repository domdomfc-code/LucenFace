"use client";

import Image from "next/image";
import { useCallback, useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const API_BASE =
  typeof process.env.NEXT_PUBLIC_API_URL === "string"
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
    : "http://localhost:8000";

type ProcessConfig = {
  ratio: "3x4" | "4x6";
  prefer_face_crop: boolean;
  replace_blue_bg: boolean;
  force_blue_despite_uniform: boolean;
  blue_hex: string;
  min_face_conf: number;
  auto_orient: boolean;
  crop_center_mode: "nose" | "face";
  letterbox_smart_framing: boolean;
  rembg_engine: "none" | "local" | "remove_bg_api";
  rembg_model: string;
};

type AuditRow = {
  work_idx: number;
  filename: string;
  status: string;
  checks: Record<string, { ok: boolean; message: string }>;
  errors: string[];
  warnings: string[];
  n_ok: number;
  n_tot: number;
};

type ProcessResultItem = {
  work_idx: number;
  filename: string;
  jpg_base64: string | null;
  download_name: string | null;
  error: string | null;
};

const defaultConfig = (): ProcessConfig => ({
  ratio: "3x4",
  prefer_face_crop: false,
  replace_blue_bg: true,
  force_blue_despite_uniform: false,
  blue_hex: "#005BC4",
  min_face_conf: 0.9,
  auto_orient: true,
  crop_center_mode: "nose",
  letterbox_smart_framing: true,
  rembg_engine: "remove_bg_api",
  rembg_model: "u2net_human_seg",
});

function buildFormData(
  files: File[],
  config: ProcessConfig,
  indices?: number[],
): FormData {
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f);
  }
  fd.append("config", JSON.stringify(config));
  if (indices !== undefined) {
    fd.append("indices", JSON.stringify(indices));
  }
  return fd;
}

export default function HomePage() {
  const [config, setConfig] = useState<ProcessConfig>(defaultConfig);
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [auditRows, setAuditRows] = useState<AuditRow[] | null>(null);
  const [rejected, setRejected] = useState<{ file: string; reason: string }[]>(
    [],
  );
  const [processPick, setProcessPick] = useState<Record<number, boolean>>({});
  const [processOut, setProcessOut] = useState<
    Record<number, ProcessResultItem>
  >({});
  const [loading, setLoading] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  const effectiveRembg = config.replace_blue_bg ? config.rembg_engine : "none";

  const refreshHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/health`);
      const j = await r.json();
      setApiOk(!!j.ok);
    } catch {
      setApiOk(false);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const onFiles = useCallback((list: FileList | File[]) => {
    const arr = Array.from(list).filter((f) => /image\/(jpeg|png)/i.test(f.type));
    setFiles(arr.slice(0, 50));
    setAuditRows(null);
    setProcessPick({});
    setProcessOut({});
    setErr(null);
  }, []);

  const runAudit = async () => {
    if (!files.length) {
      setErr("Chọn ít nhất một ảnh JPG/PNG.");
      return;
    }
    setLoading("audit");
    setErr(null);
    try {
      const cfg = { ...config, rembg_engine: effectiveRembg };
      const r = await fetch(`${API_BASE}/api/audit`, {
        method: "POST",
        body: buildFormData(files, cfg),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      const data = await r.json();
      setRejected(data.rejected || []);
      const rows: AuditRow[] = data.rows || [];
      setAuditRows(rows);
      const pick: Record<number, boolean> = {};
      for (const row of rows) {
        pick[row.work_idx] = row.status === "OK";
      }
      setProcessPick(pick);
      setProcessOut({});
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Lỗi kiểm tra.");
    } finally {
      setLoading(null);
    }
  };

  const runProcess = async () => {
    if (!files.length || !auditRows?.length) {
      setErr("Chạy bước kiểm tra trước.");
      return;
    }
    const indices = auditRows
      .map((r) => r.work_idx)
      .filter((i) => processPick[i]);
    if (!indices.length) {
      setErr("Chọn ít nhất một ảnh để xử lý.");
      return;
    }
    setLoading("process");
    setErr(null);
    try {
      const cfg = { ...config, rembg_engine: effectiveRembg };
      const r = await fetch(`${API_BASE}/api/process`, {
        method: "POST",
        body: buildFormData(files, cfg, indices),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      const data = await r.json();
      const map: Record<number, ProcessResultItem> = {};
      for (const item of data.results as ProcessResultItem[]) {
        map[item.work_idx] = item;
      }
      setProcessOut(map);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Lỗi xử lý.");
    } finally {
      setLoading(null);
    }
  };

  const downloadZip = async () => {
    if (!files.length || !auditRows?.length) {
      setErr("Chưa có dữ liệu để tải ZIP.");
      return;
    }
    const indices = auditRows
      .map((r) => r.work_idx)
      .filter((i) => processPick[i]);
    if (!indices.length) {
      setErr("Chọn ảnh cần đưa vào ZIP.");
      return;
    }
    setLoading("zip");
    setErr(null);
    try {
      const cfg = { ...config, rembg_engine: effectiveRembg };
      const r = await fetch(`${API_BASE}/api/process-zip`, {
        method: "POST",
        body: buildFormData(files, cfg, indices),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "anh_chan_dung_chuan_hoa.zip";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Không tải được ZIP.");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 pb-20 pt-6 md:px-6">
      <header className="mb-8 flex flex-col gap-4 rounded-2xl border border-cyan-500/20 bg-black p-4 shadow-[0_0_40px_-8px_rgba(0,229,255,0.25)] md:flex-row md:items-center md:justify-between md:px-5">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          {/* Khung cố định: tránh flex + min-w-0 làm next/image (w-auto) co width = 0 */}
          <div className="relative h-10 w-[min(100%,260px)] shrink-0 md:h-[52px] md:w-[min(100%,320px)]">
            <Image
              src="/lucenface-logo.png"
              alt="LucenFace — Chuẩn hóa hình ảnh khuôn mặt"
              fill
              unoptimized
              priority
              className="object-contain object-left"
              sizes="(max-width: 768px) 260px, 320px"
            />
          </div>
          <span className="hidden text-xs font-semibold uppercase tracking-wider text-cyan-400/80 md:inline">
            Web
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-cyan-500/35 bg-cyan-950/40 px-3 py-1 text-xs font-bold text-cyan-100/90">
            Batch ≤ 50 ảnh
          </span>
          <span className="rounded-full border border-cyan-500/35 bg-cyan-950/40 px-3 py-1 text-xs font-bold text-cyan-100/90">
            JPG / PNG
          </span>
          <button
            type="button"
            onClick={() => void refreshHealth()}
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-bold transition",
              apiOk === true &&
                "border-emerald-500/50 bg-emerald-950/60 text-emerald-300",
              apiOk === false &&
                "border-red-500/50 bg-red-950/50 text-red-300",
              apiOk === null &&
                "border-cyan-500/30 bg-cyan-950/30 text-cyan-200/80",
            )}
          >
            API {apiOk === true ? "OK" : apiOk === false ? "lỗi" : "…"}
          </button>
        </div>
      </header>

      <h1 className="mb-6 text-balance text-lg font-bold tracking-tight text-surface-ink md:text-2xl">
        Chuẩn hóa ảnh chân dung học sinh
      </h1>
      <p className="mb-8 max-w-3xl text-sm font-medium leading-relaxed text-surface-muted">
        Kiểm tra tiêu chí (một khuôn mặt, phông, sáng…), sau đó xử lý crop theo
        tỷ lệ 3×4 / 4×6 và tùy chọn ghép nền xanh (remove.bg hoặc rembg local).
        Ảnh chỉ gửi tới máy chủ bạn chạy API — không lưu vĩnh viễn trong app
        demo này.
      </p>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="space-y-6">
          <div
            className={cn(
              "overflow-hidden rounded-2xl border border-white/10 bg-[#1e1e24] shadow-soft transition",
              dragOver && "ring-2 ring-brand-light",
            )}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
            }}
          >
            <div className="px-6 pb-2 pt-8 text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-600">
                <svg
                  width="36"
                  height="36"
                  viewBox="0 0 56 56"
                  fill="none"
                  className="opacity-90"
                  aria-hidden
                >
                  <rect
                    x="6"
                    y="10"
                    width="44"
                    height="36"
                    rx="6"
                    fill="#4b5563"
                  />
                  <path
                    d="M12 38 L22 26 L30 34 L38 22 L44 28 V38 H12 Z"
                    fill="#9ca3af"
                  />
                  <circle cx="40" cy="18" r="5" fill="#d1d5db" />
                </svg>
              </div>
              <p className="text-sm font-semibold text-white/90">
                Kéo thả ảnh vào đây hoặc chọn tệp
              </p>
              <p className="mt-1 text-xs font-medium text-white/50">
                Tối đa 50 ảnh · ~12MB/ảnh · API:{" "}
                <code className="rounded bg-white/10 px-1">{API_BASE}</code>
              </p>
            </div>
            <div className="border-t border-dashed border-white/15 bg-[#1e1e24] px-6 py-5">
              <label className="flex cursor-pointer flex-col items-center gap-3">
                <input
                  type="file"
                  accept="image/jpeg,image/png"
                  multiple
                  className="hidden"
                  onChange={(e) =>
                    e.target.files?.length && onFiles(e.target.files)
                  }
                />
                <span className="rounded-full bg-brand px-6 py-2.5 text-sm font-bold text-white shadow-glow transition hover:brightness-110">
                  + Chọn ảnh
                </span>
              </label>
            </div>
          </div>

          {files.length > 0 && (
            <p className="text-sm font-semibold text-surface-ink">
              Đã chọn{" "}
              <span className="text-brand">{files.length}</span> ảnh
            </p>
          )}

          {err && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-900">
              {err}
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              disabled={!files.length || loading !== null}
              onClick={() => void runAudit()}
              className="rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-sm font-bold text-surface-ink shadow-sm transition hover:bg-slate-50 disabled:opacity-50"
            >
              {loading === "audit" ? "Đang kiểm tra…" : "1 · Kiểm tra ảnh"}
            </button>
            <button
              type="button"
              disabled={
                !auditRows?.length || !files.length || loading !== null
              }
              onClick={() => void runProcess()}
              className="rounded-xl bg-brand px-6 py-2.5 text-sm font-bold text-white shadow-glow transition hover:brightness-110 disabled:opacity-50"
            >
              {loading === "process" ? "Đang xử lý…" : "2 · Xử lý đã chọn"}
            </button>
            <button
              type="button"
              disabled={!auditRows?.length || loading !== null}
              onClick={() => void downloadZip()}
              className="rounded-xl border border-brand/30 bg-brand/5 px-5 py-2.5 text-sm font-bold text-brand transition hover:bg-brand/10 disabled:opacity-50"
            >
              {loading === "zip" ? "ZIP…" : "Tải ZIP"}
            </button>
          </div>

          {rejected.length > 0 && (
            <div className="rounded-2xl border border-amber-200 bg-amber-50/80 p-4">
              <p className="mb-2 text-sm font-bold text-amber-900">
                Một số file bị loại
              </p>
              <ul className="max-h-40 space-y-1 overflow-auto text-xs font-medium text-amber-950">
                {rejected.map((x) => (
                  <li key={x.file}>
                    <span className="font-mono">{x.file}</span> — {x.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {auditRows && auditRows.length > 0 && (
            <div className="space-y-6">
              <h2 className="text-base font-bold text-surface-ink">
                Kết quả kiểm tra
              </h2>
              {auditRows.map((row) => (
                <article
                  key={row.work_idx}
                  className="rounded-2xl border border-slate-200/90 bg-white/90 p-4 shadow-soft"
                >
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="font-mono text-sm font-bold text-surface-ink">
                        {row.filename}
                      </p>
                      <p className="text-xs font-semibold text-surface-muted">
                        Checklist {row.n_ok}/{row.n_tot} ·{" "}
                        <span
                          className={
                            row.status === "OK"
                              ? "text-emerald-700"
                              : "text-red-700"
                          }
                        >
                          {row.status}
                        </span>
                      </p>
                    </div>
                    <label className="flex cursor-pointer items-center gap-2 text-sm font-bold text-surface-ink">
                      <input
                        type="checkbox"
                        checked={!!processPick[row.work_idx]}
                        onChange={(e) =>
                          setProcessPick((p) => ({
                            ...p,
                            [row.work_idx]: e.target.checked,
                          }))
                        }
                      />
                      Xử lý bước 2
                    </label>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <p className="mb-2 text-xs font-bold uppercase tracking-wide text-surface-muted">
                        Tiêu chí
                      </p>
                      <div className="overflow-hidden rounded-xl border border-slate-100">
                        <table className="w-full text-left text-xs">
                          <thead className="bg-slate-50 font-bold text-surface-muted">
                            <tr>
                              <th className="px-3 py-2">Đạt</th>
                              <th className="px-3 py-2">Mục</th>
                              <th className="px-3 py-2">Chi tiết</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(row.checks).map(([k, v]) => (
                              <tr
                                key={k}
                                className="border-t border-slate-100 bg-white"
                              >
                                <td className="px-3 py-2 font-bold">
                                  {v.ok ? "Có" : "Không"}
                                </td>
                                <td className="px-3 py-2 font-semibold">
                                  {k}
                                </td>
                                <td className="px-3 py-2 text-surface-muted">
                                  {v.message}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {(row.errors.length > 0 || row.warnings.length > 0) && (
                        <div className="mt-2 space-y-1 text-xs">
                          {row.errors.map((e) => (
                            <p key={e} className="font-medium text-red-700">
                              {e}
                            </p>
                          ))}
                          {row.warnings.map((w) => (
                            <p key={w} className="font-medium text-amber-800">
                              {w}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                    <div>
                      <p className="mb-2 text-xs font-bold uppercase tracking-wide text-surface-muted">
                        Ảnh sau xử lý
                      </p>
                      {processOut[row.work_idx]?.jpg_base64 ? (
                        <div className="space-y-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`data:image/jpeg;base64,${processOut[row.work_idx].jpg_base64}`}
                            alt="Processed"
                            className="w-full max-w-xs rounded-xl border border-slate-200 object-contain"
                          />
                          <a
                            className="inline-flex rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white hover:bg-slate-800"
                            href={`data:image/jpeg;base64,${processOut[row.work_idx].jpg_base64}`}
                            download={
                              processOut[row.work_idx].download_name ||
                              "out.jpg"
                            }
                          >
                            Tải JPG
                          </a>
                        </div>
                      ) : processOut[row.work_idx]?.error ? (
                        <p className="text-sm font-medium text-red-700">
                          {processOut[row.work_idx].error}
                        </p>
                      ) : (
                        <p className="text-sm font-medium text-surface-muted">
                          Chạy bước 2 để xem ảnh đã xử lý.
                        </p>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <aside className="h-fit space-y-4 rounded-2xl border border-slate-200/90 bg-white/90 p-5 shadow-soft lg:sticky lg:top-6">
          <h2 className="text-sm font-bold text-surface-ink">Cấu hình</h2>

          <label className="block text-xs font-bold text-surface-muted">
            Tỷ lệ đầu ra
            <select
              className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold"
              value={config.ratio}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  ratio: e.target.value as ProcessConfig["ratio"],
                }))
              }
            >
              <option value="3x4">3 × 4</option>
              <option value="4x6">4 × 6</option>
            </select>
          </label>

          <Toggle
            label="Cắt theo khuôn mặt"
            checked={config.prefer_face_crop}
            onChange={(v) => setConfig((c) => ({ ...c, prefer_face_crop: v }))}
          />
          <Toggle
            label="Ghép nền xanh / rembg"
            checked={config.replace_blue_bg}
            onChange={(v) => setConfig((c) => ({ ...c, replace_blue_bg: v }))}
          />
          {config.replace_blue_bg && (
            <>
              <Toggle
                label="Luôn ghép nền (kể cả phông một màu)"
                checked={config.force_blue_despite_uniform}
                onChange={(v) =>
                  setConfig((c) => ({
                    ...c,
                    force_blue_despite_uniform: v,
                  }))
                }
              />
              <label className="block text-xs font-bold text-surface-muted">
                Màu nền
                <input
                  type="color"
                  className="mt-1 h-10 w-full cursor-pointer rounded-xl border border-slate-200"
                  value={config.blue_hex}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, blue_hex: e.target.value }))
                  }
                />
              </label>
              <label className="block text-xs font-bold text-surface-muted">
                Engine tách nền
                <select
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold"
                  value={config.rembg_engine}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      rembg_engine: e.target
                        .value as ProcessConfig["rembg_engine"],
                    }))
                  }
                >
                  <option value="remove_bg_api">remove.bg (API)</option>
                  <option value="local">rembg local</option>
                </select>
              </label>
              {config.rembg_engine === "local" && (
                <label className="block text-xs font-bold text-surface-muted">
                  Model ONNX
                  <select
                    className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold"
                    value={config.rembg_model}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        rembg_model: e.target.value,
                      }))
                    }
                  >
                    <option value="u2net_human_seg">u2net_human_seg</option>
                    <option value="u2net">u2net</option>
                    <option value="isnet-general-use">isnet-general-use</option>
                    <option value="silueta">silueta</option>
                  </select>
                </label>
              )}
            </>
          )}

          <label className="block text-xs font-bold text-surface-muted">
            Độ tin cậy phát hiện mặt ({config.min_face_conf.toFixed(2)})
            <input
              type="range"
              min={0.3}
              max={0.9}
              step={0.05}
              className="mt-2 w-full"
              value={config.min_face_conf}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  min_face_conf: Number(e.target.value),
                }))
              }
            />
          </label>

          <Toggle
            label="Kiểm tra hướng ảnh (xoay tự động)"
            checked={config.auto_orient}
            onChange={(v) => setConfig((c) => ({ ...c, auto_orient: v }))}
          />

          <label className="block text-xs font-bold text-surface-muted">
            Căn tâm khi crop
            <select
              className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold"
              value={config.crop_center_mode}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  crop_center_mode: e.target.value as "nose" | "face",
                }))
              }
            >
              <option value="nose">Mũi (MediaPipe)</option>
              <option value="face">Tâm bbox mặt</option>
            </select>
          </label>

          <Toggle
            label="Viền letterbox thông minh"
            checked={config.letterbox_smart_framing}
            onChange={(v) =>
              setConfig((c) => ({ ...c, letterbox_smart_framing: v }))
            }
          />

          <p className="border-t border-slate-100 pt-4 text-xs leading-relaxed text-surface-muted">
            remove.bg: đặt biến môi trường{" "}
            <code className="rounded bg-slate-100 px-1">REMOVEBG_API_KEY</code>{" "}
            trên máy chạy API.
          </p>
        </aside>
      </div>
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2.5">
      <span className="text-sm font-bold text-surface-ink">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative h-7 w-12 rounded-full transition",
          checked ? "bg-brand" : "bg-slate-300",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-6 w-6 rounded-full bg-white shadow transition",
            checked ? "left-5" : "left-0.5",
          )}
        />
      </button>
    </label>
  );
}
