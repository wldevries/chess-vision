// Single frame (webcam snap or file) -> FEN, fully in-browser.
//   corners.onnx (512) -> 81 lattice pts -> homographyFromLattice (geometry.js)
//   board-crop -> pieces.onnx (1280 letterbox) -> per-piece class + contact keypoint
//   contact pt -> H -> square -> FEN (+ 4-way orientation, lichess link, unicode board).
// onnxruntime-web (global `ort`, WebGPU bundle) does inference; geometry.js does the
// deterministic geometry (verified vs Python). Decode/NMS match scripts/diag_pose_onnx.py.

import {
  homographyFromLattice,
  squareForPoint,
  rotateBoard,
  projectLattice,
  FILES,
} from "./geometry.js";

const CORNER_SIZE = 512;
const POSE_SIZE = 1280;
const CONF = 0.25;
const IOU = 0.45;
const PAD = 114;
const N_ANCHORS = 33600;
const PIECE_FEN = "PRNBQKprnbqk"; // class id 0..11 -> FEN letter
const UNICODE = { P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔", p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚" };
const CROP = { side: 0.12, top: 0.3, bottom: 0.08 };
const TARGET_AR = 4 / 3; // camera frame is centre-cropped to this (matches #cam in index.html)

const $ = (id) => document.getElementById(id);
const status = (m) => ($("status").textContent = m);

let cornerSess = null, pieceSess = null;
let forcedWasm = false; // set true after an accelerator op fails at run time
let boardR0 = null, rotation = 0;
let scene = null; // {src, H, dets, crop, scale, padX, padY} kept for redraw on rotate

const IS_MOBILE = navigator.userAgentData?.mobile
  ?? /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
// The mobile WebGPU mis-compute is Chromium-only (Chrome/Edge/Samsung); Firefox mobile WebGPU is fine.
const IS_CHROMIUM_MOBILE = IS_MOBILE
  && /Chrome|Chromium|CriOS|Edg|SamsungBrowser/.test(navigator.userAgent)
  && !/Firefox|FxiOS/.test(navigator.userAgent);

// Execution-provider list for the chosen backend (always WASM-terminated as a safety net).
// WebNN partitions the graph and runs unsupported nodes on CPU, so it degrades rather than fails.
function epList() {
  if (forcedWasm) return ["wasm"];
  const webnn = (deviceType) => ({ name: "webnn", deviceType, powerPreference: "high-performance" });
  switch (document.getElementById("backend").value) {
    case "webgpu": return ["webgpu", "wasm"];
    case "webnn-gpu": return [webnn("gpu"), "wasm"];
    case "webnn-npu": return [webnn("npu"), "wasm"];
    case "wasm": return ["wasm"];
    default: // "auto": WebGPU is reliable on desktop AND on Firefox mobile, but Chromium MOBILE
             // WebGPU SILENTLY mis-computes the corner model (soft-argmax collapses -> all 81 lattice
             // pts pile up at image centre => "tiny board"). It's wrong results, not an error, so the
             // runtime fallback never fires. So on Chromium-mobile only, route around WebGPU via
             // WebNN (npu, then gpu), then WASM.
      return IS_CHROMIUM_MOBILE ? [webnn("npu"), webnn("gpu"), "wasm"] : ["webgpu", "wasm"];
  }
}

async function createSessions() {
  // Release any existing sessions first, and create SEQUENTIALLY: ort-web's WebGPU EP forbids
  // creating two sessions concurrently ("another WebGPU EP inference session is being created").
  for (const s of [cornerSess, pieceSess]) { try { await s?.release?.(); } catch { /* ignore */ } }
  cornerSess = pieceSess = null;
  try {
    const opt = { executionProviders: epList() };
    cornerSess = await ort.InferenceSession.create("./models/corners.onnx", opt);
    pieceSess = await ort.InferenceSession.create("./models/pieces.onnx", opt);
  } catch (e) {
    if (forcedWasm) throw e; // already on the CPU safety net -> genuinely broken
    forcedWasm = true;       // a backend failed to even create -> drop to WASM and rebuild
    return createSessions();
  }
}

const backendLabel = () => (forcedWasm ? "wasm (fallback)" : document.getElementById("backend").value);

async function loadModels() {
  try {
    await createSessions();
    status(`models ready — backend ${backendLabel()}. Start the camera or pick a file.`);
  } catch (e) {
    // Some backends (e.g. WebNN on the Cast op) fail at CREATE time, not run time. Drop to WASM.
    if (!forcedWasm) {
      status(`${backendLabel()} unavailable (${e}) — falling back to WASM…`);
      forcedWasm = true;
      try {
        await createSessions();
        status("models ready — backend wasm (fallback). Start the camera or pick a file.");
      } catch (e2) {
        status("failed to load models: " + e2);
      }
    } else status("failed to load models: " + e);
  }
}

// WebGPU/WebNN kernels can fail only at RUN time (e.g. an fp16 op gap). Detect that, drop to
// WASM for the rest of the session, and let the caller retry once.
function isAcceleratorOpError(e) {
  return !forcedWasm && /webgpu|webnn|kernel|data type|not implemented|unsupported/i.test(String(e));
}
async function fallbackToWasm() {
  forcedWasm = true;
  cornerSess = pieceSess = null;
  await createSessions();
}

// ---- preprocessing (source = any drawable: <canvas>/<img>) -------------------

function toCHW(ctx, size) {
  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size, chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    chw[p] = data[p * 4] / 255;
    chw[plane + p] = data[p * 4 + 1] / 255;
    chw[2 * plane + p] = data[p * 4 + 2] / 255;
  }
  return chw;
}

function tmpCtx(size) {
  const cv = document.createElement("canvas");
  cv.width = size; cv.height = size;
  return cv.getContext("2d", { willReadFrequently: true });
}

async function detectCorners(src, w, h) {
  // predict_corners resize-STRETCHES to a square; normalized outputs scale back by w/h.
  const ctx = tmpCtx(CORNER_SIZE);
  ctx.drawImage(src, 0, 0, w, h, 0, 0, CORNER_SIZE, CORNER_SIZE);
  const chw = toCHW(ctx, CORNER_SIZE);
  const out = await cornerSess.run({
    [cornerSess.inputNames[0]]: new ort.Tensor("float32", chw, [1, 3, CORNER_SIZE, CORNER_SIZE]),
  });
  const c = out[cornerSess.outputNames[0]].data;
  const pts = [];
  for (let i = 0; i < 81; i++) pts.push([c[i * 2] * w, c[i * 2 + 1] * h]);
  return pts;
}

function boardCropBbox(pts, w, h) {
  let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
  for (const [x, y] of pts) { a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y); }
  const bw = c - a, bh = d - b;
  return {
    x0: Math.max(0, Math.floor(a - CROP.side * bw)),
    y0: Math.max(0, Math.floor(b - CROP.top * bh)),
    x1: Math.min(w, Math.floor(c + CROP.side * bw)),
    y1: Math.min(h, Math.floor(d + CROP.bottom * bh)),
  };
}

async function detectPieces(src, crop) {
  const cw = crop.x1 - crop.x0, ch = crop.y1 - crop.y0;
  const scale = Math.min(POSE_SIZE / cw, POSE_SIZE / ch);
  const nw = Math.round(cw * scale), nh = Math.round(ch * scale);
  const padX = Math.floor((POSE_SIZE - nw) / 2), padY = Math.floor((POSE_SIZE - nh) / 2);
  const ctx = tmpCtx(POSE_SIZE);
  ctx.fillStyle = `rgb(${PAD},${PAD},${PAD})`;
  ctx.fillRect(0, 0, POSE_SIZE, POSE_SIZE);
  ctx.drawImage(src, crop.x0, crop.y0, cw, ch, padX, padY, nw, nh);
  const chw = toCHW(ctx, POSE_SIZE);
  const out = await pieceSess.run({
    [pieceSess.inputNames[0]]: new ort.Tensor("float32", chw, [1, 3, POSE_SIZE, POSE_SIZE]),
  });
  return { dets: decodePose(out[pieceSess.outputNames[0]].data), scale, padX, padY };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

// output0 [1,19,33600] channel-major: [0..3]=cx,cy,w,h [4..15]=12 cls(prob) [16..18]=kx,ky,kconf.
function decodePose(data) {
  const dets = [];
  for (let a = 0; a < N_ANCHORS; a++) {
    let best = 0, cls = 0;
    for (let k = 0; k < 12; k++) {
      const s = data[(4 + k) * N_ANCHORS + a];
      if (s > best) { best = s; cls = k; }
    }
    if (best < CONF) continue;
    const cx = data[a], cy = data[N_ANCHORS + a];
    const w = data[2 * N_ANCHORS + a], h = data[3 * N_ANCHORS + a];
    dets.push({
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
      score: best, cls, kx: data[16 * N_ANCHORS + a], ky: data[17 * N_ANCHORS + a],
    });
  }
  dets.sort((p, q) => q.score - p.score);
  const keep = [];
  for (const d of dets) if (keep.every((k) => iou(d, k) < IOU)) keep.push(d);
  return keep;
}

// keypoint (1280 letterbox) -> crop frame -> full-frame pixel.
const kpToFull = (d, crop, scale, padX, padY) => [
  (d.kx - padX) / scale + crop.x0,
  (d.ky - padY) / scale + crop.y0,
];

// ---- rendering ---------------------------------------------------------------

function fenBoard(board) {
  const rows = [];
  for (let r = 8; r >= 1; r--) {
    let row = "", empty = 0;
    for (let f = 0; f < 8; f++) {
      const sq = `${FILES[f]}${r}`;
      if (sq in board) { if (empty) { row += empty; empty = 0; } row += PIECE_FEN[board[sq]]; }
      else empty++;
    }
    if (empty) row += empty;
    rows.push(row);
  }
  return rows.join("/");
}

// Render the board as a fixed 8x8 grid of cells (NOT text -- chess glyphs and '·' have
// different advance widths, so a monospace text grid never aligns).
function renderBoardGrid(board) {
  const el = $("board");
  el.textContent = "";
  for (let r = 8; r >= 1; r--) {
    for (let f = 0; f < 8; f++) {
      const sq = `${FILES[f]}${r}`;
      const cell = document.createElement("div");
      const dark = (f + r) % 2 === 1; // a1 (0+1) dark, a8 light -- real board colours
      cell.className = "sq " + (dark ? "dark" : "light");
      if (sq in board) {
        const cls = board[sq];
        cell.classList.add(cls < 6 ? "w" : "b");
        const g = document.createElement("span");
        g.textContent = UNICODE[PIECE_FEN[cls]];
        cell.appendChild(g); // span so coord labels below aren't wiped by textContent
      }
      // Coordinates (lichess-style, opposite-tone): ranks down the a-file, files along rank 1.
      const tone = dark ? "#eccfa5" : "#b58863";
      if (f === 0) cell.appendChild(coordLabel("rank", r, tone));
      if (r === 1) cell.appendChild(coordLabel("file", FILES[f], tone));
      el.appendChild(cell);
    }
  }
}

function coordLabel(kind, text, color) {
  const s = document.createElement("span");
  s.className = "coord " + kind;
  s.textContent = text;
  s.style.color = color;
  return s;
}

function drawScene() {
  if (!scene) return;
  const { src, H, dets, crop, scale, padX, padY } = scene;
  const w = src.width, h = src.height;
  const cv = $("overlay");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.drawImage(src, 0, 0); // frozen frame as base

  // Board grid (project the 9x9 lattice through H).
  const grid = projectLattice(H);
  ctx.strokeStyle = "rgba(40,220,120,0.9)";
  ctx.lineWidth = Math.max(1.5, w / 700);
  ctx.beginPath();
  for (let j = 0; j < 9; j++) { ctx.moveTo(...grid[j][0]); for (let i = 1; i < 9; i++) ctx.lineTo(...grid[j][i]); }
  for (let i = 0; i < 9; i++) { ctx.moveTo(...grid[0][i]); for (let j = 1; j < 9; j++) ctx.lineTo(...grid[j][i]); }
  ctx.stroke();

  // Contact points + piece letters.
  ctx.font = `${Math.round(w / 45)}px sans-serif`;
  ctx.textAlign = "center";
  ctx.lineWidth = 3;
  for (const d of dets) {
    const [fx, fy] = kpToFull(d, crop, scale, padX, padY);
    ctx.fillStyle = d.cls < 6 ? "#1e90ff" : "#ff3b30";
    ctx.beginPath(); ctx.arc(fx, fy, w / 200, 0, 7); ctx.fill();
    ctx.strokeStyle = "#000"; ctx.strokeText(PIECE_FEN[d.cls], fx, fy - w / 90);
    ctx.fillStyle = "#fff"; ctx.fillText(PIECE_FEN[d.cls], fx, fy - w / 90);
  }
  // Show the frozen annotated frame: drop the 4:3 crop so a tall still isn't clipped, let the
  // overlay flow (height by its own aspect), and hide the video.
  enterStillMode();
}

// Switch the stage from the live video to the in-flow still canvas (shared by the plain frozen
// frame and the annotated result).
function enterStillMode() {
  $("stage").classList.remove("live"); // drop the 4:3 crop so a tall still isn't clipped
  $("cam").style.display = "none";
  const ov = $("overlay");
  ov.style.position = "relative";
  ov.style.height = "auto";
}

// Freeze the captured frame immediately (before the multi-second inference) so the user sees the
// shot they took, not the still-live feed. drawScene later repaints the same canvas with overlays.
function showFrozen(src) {
  const ov = $("overlay");
  ov.width = src.width; ov.height = src.height;
  ov.getContext("2d").drawImage(src, 0, 0);
  enterStillMode();
}

const setBusy = (on) => { $("busy").hidden = !on; };

function showLive() {
  setBusy(false);
  $("stage").classList.add("live"); // container 4:3 -> object-fit:cover centre-crops the video
  $("cam").style.display = "block";
  const ov = $("overlay");
  ov.style.position = "absolute";
  ov.style.height = "100%";
  ov.getContext("2d").clearRect(0, 0, ov.width, ov.height);
}

function renderResult() {
  const b = rotateBoard(boardR0, rotation);
  const fen = `${fenBoard(b)} w - - 0 1`;
  $("fen").textContent = fen;
  $("orient").textContent = rotation;
  renderBoardGrid(b);
  // Lichess editor wants literal '/' rank separators and '_' for spaces (NOT %2F/%20).
  $("lichess").href = `https://lichess.org/editor/${fen.replaceAll(" ", "_")}`;
}

// ---- pipeline ----------------------------------------------------------------

async function readFrame(src, w, h) {
  if (!cornerSess || !pieceSess) return status("models not loaded — pick another backend or reload.");
  $("snap").disabled = true;
  showFrozen(src); // freeze the shot immediately; show the spinner over it during inference
  setBusy(true);
  try {
    status(`detecting board… (${backendLabel()})`);
    const pts = await detectCorners(src, w, h);
    const H = homographyFromLattice(pts);
    status("detecting pieces…");
    const crop = boardCropBbox(pts, w, h);
    const { dets, scale, padX, padY } = await detectPieces(src, crop);

    const bySquare = {};
    for (const d of dets) {
      const sq = squareForPoint(H, kpToFull(d, crop, scale, padX, padY));
      if (!sq) continue;
      if (!(sq in bySquare) || d.score > bySquare[sq].score) bySquare[sq] = d;
    }
    boardR0 = {};
    for (const [sq, d] of Object.entries(bySquare)) boardR0[sq] = d.cls;
    rotation = 0;
    scene = { src, H, dets, crop, scale, padX, padY };
    drawScene();
    renderResult();
    setBusy(false);
    $("rotate").disabled = false;
    $("resume").disabled = false;
    status(`${dets.length} pieces on ${Object.keys(boardR0).length} squares (${backendLabel()}).`);
  } catch (e) {
    if (isAcceleratorOpError(e)) {
      status(`${backendLabel()} op unsupported — switching to WASM and retrying…`);
      await fallbackToWasm();
      $("snap").disabled = false;
      return readFrame(src, w, h); // retry once on WASM (it re-shows its own busy spinner)
    }
    setBusy(false);
    status("error: " + e + "\n" + (e.stack || ""));
  } finally {
    $("snap").disabled = false;
  }
}

// Centre-crop a drawable to aspect `ar` (cover semantics), mirroring the CSS object-fit:cover
// preview so what the model sees == what the user framed. Trained corners are square-to-landscape
// (w/h in ~[1.0, 1.5]); a raw portrait phone frame is out-of-distribution and detects a tiny board.
function cropToAspect(src, sw, sh, ar) {
  let cw = sw, ch = sh;
  if (sw / sh > ar) cw = Math.round(sh * ar); // too wide -> trim sides
  else ch = Math.round(sw / ar);              // too tall -> trim top/bottom
  const sx = Math.floor((sw - cw) / 2), sy = Math.floor((sh - ch) / 2);
  const cv = document.createElement("canvas");
  cv.width = cw; cv.height = ch;
  cv.getContext("2d").drawImage(src, sx, sy, cw, ch, 0, 0, cw, ch);
  return cv;
}

function frameFromVideo() {
  const v = $("cam");
  if (!v.videoWidth) return document.createElement("canvas"); // width 0 -> snap handler bails
  return cropToAspect(v, v.videoWidth, v.videoHeight, TARGET_AR);
}

// ---- camera + inputs ---------------------------------------------------------

let currentStream = null;

// Phones expose several rear cameras (main / ultra-wide / tele / depth) and
// facingMode:"environment" often hands back the ultra-wide. Score the labels so
// the main lens wins by default; the user can still override via the dropdown.
function preferredCameraId(cams) {
  if (!cams.length) return null;
  const score = (label) => {
    const l = label.toLowerCase();
    if (/ultra|wide|tele|macro|depth|mono|ir\b|front|facing front|user/.test(l)) return -1;
    return 1;
  };
  const ranked = [...cams].sort((a, b) => score(b.label) - score(a.label));
  return ranked[0].deviceId;
}

async function populateCameras(activeId) {
  let devices = [];
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { return; }
  const cams = devices.filter((d) => d.kind === "videoinput");
  // Labels are only populated once permission is granted; bail until then.
  if (!cams.length || !cams.some((c) => c.label)) return;
  const sel = $("camera");
  sel.innerHTML = "";
  cams.forEach((c, i) => {
    const opt = document.createElement("option");
    opt.value = c.deviceId;
    opt.textContent = c.label || `Camera ${i + 1}`;
    sel.appendChild(opt);
  });
  if (activeId) sel.value = activeId;
  $("camWrap").style.display = cams.length > 1 ? "" : "none";
}

async function startCamera(deviceId) {
  try {
    for (const t of currentStream?.getTracks?.() || []) t.stop();
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } };
    const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    currentStream = stream;
    const v = $("cam");
    v.srcObject = stream;
    await v.play();
    showLive();
    $("snap").disabled = false;
    status("camera live — frame the board and Snap.");

    const activeId = stream.getVideoTracks()[0]?.getSettings?.().deviceId;
    await populateCameras(activeId);
    // First start (no explicit pick): if the phone gave us a non-main lens but a
    // better one exists, switch to it once.
    if (!deviceId) {
      const cams = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "videoinput" && d.label);
      const preferred = preferredCameraId(cams);
      if (preferred && preferred !== activeId) return startCamera(preferred);
    }
  } catch (e) {
    status("camera error: " + e + " (needs https or localhost + permission)");
  }
}

$("start").addEventListener("click", () => startCamera());
$("camera").addEventListener("change", (e) => startCamera(e.target.value));
$("resume").addEventListener("click", () => {
  showLive();
  status("camera live.");
});
$("snap").addEventListener("click", () => {
  const f = frameFromVideo();
  if (!f.width) return status("no camera frame yet.");
  readFrame(f, f.width, f.height);
});
$("file").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const cv = document.createElement("canvas");
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    cv.getContext("2d").drawImage(img, 0, 0);
    readFrame(cv, cv.width, cv.height);
  };
  img.src = URL.createObjectURL(file);
});
$("rotate").addEventListener("click", () => { rotation = (rotation + 1) % 4; renderResult(); });
$("backend").addEventListener("change", async () => {
  forcedWasm = false; // createSessions releases the old sessions itself
  status(`switching backend to ${backendLabel()}…`);
  await loadModels();
});

renderBoardGrid({}); // show an empty, labelled board on load instead of blank space
loadModels();
