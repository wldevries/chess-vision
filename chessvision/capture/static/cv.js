/* Shared front-end for the chessvision capture app.
 *
 * The capture, live-read, and position-labelling pages all share the same machinery:
 * fetch helpers, a toast, object-fit:contain coordinate mapping, a webcam picker, the
 * perspective grid math, and the 4-corner marking modal. This file is that common
 * core, loaded as a plain <script> before each page's own script so everything lives
 * in one global scope (no bundler). Page scripts override the small hooks documented
 * below (`onCameraStarted`) and drive the corner modal through the `CM` controller.
 */

const $ = (id) => document.getElementById(id);
const ORIENTATIONS = ["R0", "R90", "R180", "R270"];

/* ---------- fetch helpers ---------- */
async function j(r) { const d = await r.json(); if (!r.ok) throw new Error(d.error || r.statusText); return d; }
function postForm(url, fields) {
  const body = new FormData();
  for (const [k, v] of Object.entries(fields)) body.append(k, v);
  return fetch(url, { method: "POST", body }).then(j);
}
function postJson(url, obj) {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) }).then(j);
}

/* ---------- toast ---------- */
let _toastTimer = null;
function toast(msg) {
  const t = $("toast"); if (!t) return;
  t.textContent = msg; t.classList.add("show");
  clearTimeout(_toastTimer); _toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- coordinate mapping for object-fit: contain ---------- */
// A video/image drawn with object-fit:contain is centred and letterboxed inside its
// box; these convert between that displayed frame and the source's native pixels.
function containRect(vw, vh, cw, ch) {
  const scale = Math.min(cw / vw, ch / vh);
  return { scale, offx: (cw - vw * scale) / 2, offy: (ch - vh * scale) / 2 };
}
const toDisplay = (x, y, r) => [r.offx + x * r.scale, r.offy + y * r.scale];
const toNative = (x, y, r) => [(x - r.offx) / r.scale, (y - r.offy) / r.scale];

/* ---------- perspective grid math ---------- */
// Sort 4 points into TL/TR/BR/BL (mirrors geometry.order_corners): split by y into the
// top/bottom pair, then each pair by x. Click order doesn't matter.
function orderCorners(pts) {
  const s = pts.slice().sort((a, b) => a[1] - b[1]);
  const [tl, tr] = s.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const [bl, br] = s.slice(2, 4).sort((a, b) => a[0] - b[0]);
  return { tl, tr, br, bl };
}
// Homography mapping the unit square (0,0)(1,0)(1,1)(0,1) -> tl,tr,br,bl (Heckbert
// square->quad). Lets us draw the perspective-correct 8x8 grid from just the 4 corners.
function squareToQuad(tl, tr, br, bl) {
  const [x0, y0] = tl, [x1, y1] = tr, [x2, y2] = br, [x3, y3] = bl;
  const dx1 = x1 - x2, dx2 = x3 - x2, sx = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, sy = y0 - y1 + y2 - y3;
  if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {  // affine (parallelogram)
    return { a: x1 - x0, b: x3 - x0, c: x0, d: y1 - y0, e: y3 - y0, f: y0, g: 0, h: 0 };
  }
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (sx * dy2 - dx2 * sy) / den, h = (dx1 * sy - sx * dy1) / den;
  return {
    a: x1 - x0 + g * x1, b: x3 - x0 + h * x3, c: x0,
    d: y1 - y0 + g * y1, e: y3 - y0 + h * y3, f: y0, g, h,
  };
}
function projectH(H, u, v) {
  const w = H.g * u + H.h * v + 1;
  return [(H.a * u + H.b * v + H.c) / w, (H.d * u + H.e * v + H.f) / w];
}

/* ---------- overlay drawing (grid + base dots), shared by capture & live ---------- */
function drawGrid(ctx, lat, r) {
  const P = (k) => toDisplay(lat[k][0], lat[k][1], r);
  ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(79,140,255,.55)";
  for (let row = 0; row < 9; row++) {
    ctx.beginPath();
    for (let col = 0; col < 9; col++) { const [x, y] = P(row * 9 + col); col ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  }
  for (let col = 0; col < 9; col++) {
    ctx.beginPath();
    for (let row = 0; row < 9; row++) { const [x, y] = P(row * 9 + col); row ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  }
}
function drawDot(ctx, x, y, color, r0 = 6) {
  ctx.beginPath(); ctx.arc(x, y, r0, 0, 7);
  ctx.fillStyle = color === "w" ? "#f5f5f5" : "#1a1a1a";
  ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = color === "w" ? "#1a1a1a" : "#f5f5f5"; ctx.stroke();
}
// The overlay canvas context + its contain-rect, or null while the corner modal is open
// or the video has no frame yet (also clears the canvas in those cases).
function overlayCtx() {
  const c = $("overlayCanvas"), v = $("video");
  if (!c || !v) return null;
  if (CM.active || !v.videoWidth) { c.width = c.width; return null; }
  c.width = c.clientWidth; c.height = c.clientHeight;
  return { ctx: c.getContext("2d"), r: containRect(v.videoWidth, v.videoHeight, c.width, c.height) };
}

/* ---------- camera ---------- */
let stream = null;
async function listCameras() {
  try {
    if (!stream) {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach((t) => t.stop());
    }
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const sel = $("cameraSelect");
    sel.innerHTML = "";
    cams.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
      sel.appendChild(o);
    });
    const saved = localStorage.getItem("cv_camera");
    if (saved && cams.some((d) => d.deviceId === saved)) sel.value = saved;
    if (cams.length) await startCamera(sel.value);
    else toast("No camera found");
  } catch (e) { toast("Camera blocked: " + e.message); }
}
async function startCamera(deviceId) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  // 1080p, not 4K: the detector downscales every frame to <=1333px on the long side
  // (model min_size=800/max_size=1333, training data capped at 1333), so 4K gains zero
  // accuracy while many webcams only deliver it at ~1 fps. 1920 keeps preview smooth.
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 },
    },
  });
  $("video").srcObject = stream;
  if ($("camEmpty")) $("camEmpty").style.display = "none";
  localStorage.setItem("cv_camera", deviceId);
  if (typeof window.onCameraStarted === "function") window.onCameraStarted();
}
function currentDeviceId() { return $("cameraSelect").value; }
// A JPEG blob of the current video frame (uses the hidden #canvas as scratch).
function captureBlob() {
  const v = $("video"), c = $("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
}

/* ---------- corner-marking modal (CM) ---------- */
// One controller for the shared #cornerModal. A page opens it on a frozen frame (a
// canvas snapshot of the video, or a loaded photo), the user taps/drag the 4 corners
// over a live perspective grid, and the page's `onSave` callback receives the ordered
// corners. Optionally pre-fills handles from a corner-regressor prediction.
const CM = {
  active: false,
  pts: [],          // clicked points (native px of `frozen`), any order
  frozen: null,     // offscreen canvas being marked
  dragging: null,   // index of the corner being dragged, or null
  opts: {},         // { initialPts, onSave(ordered, raw), autoPredict, tagProgress }
  predictAvailable: false,

  // Probe whether the corner regressor is wired (--corner-ckpt). Returns the raw
  // availability object (also carries `heatmap` for the live page) and toggles the
  // Predict button. Safe to call when the button is absent.
  async checkAssist() {
    let info = { available: false };
    try { info = await fetch("/api/corners/available").then(j); } catch { /* off */ }
    this.predictAvailable = !!info.available;
    if ($("predictCorner")) $("predictCorner").hidden = !this.predictAvailable;
    return info;
  },

  open(frozen, opts = {}) {
    this.frozen = frozen;
    this.opts = opts;
    this.pts = (opts.initialPts && opts.initialPts.length === 4) ? opts.initialPts.map((p) => p.slice()) : [];
    this.dragging = null;
    this.active = true;
    const tp = $("taggingProgress");
    if (tp) { if (opts.tagProgress) { tp.hidden = false; tp.innerHTML = opts.tagProgress; } else { tp.hidden = true; } }
    $("cornerModal").classList.add("show");
    requestAnimationFrame(() => this.redraw());
    if (opts.autoPredict && this.predictAvailable && this.pts.length !== 4) this.predict();
  },

  close() {
    this.active = false; this.pts = []; this.frozen = null; this.dragging = null; this.opts = {};
    $("cornerModal").classList.remove("show");
  },

  _rect() {
    const c = $("cornerCanvas");
    return containRect(this.frozen.width, this.frozen.height, c.clientWidth, c.clientHeight);
  },
  _toNative(ev) {
    const c = $("cornerCanvas"), rect = c.getBoundingClientRect(), r = this._rect();
    const [nx, ny] = toNative(ev.clientX - rect.left, ev.clientY - rect.top, r);
    return [Math.max(0, Math.min(this.frozen.width, nx)), Math.max(0, Math.min(this.frozen.height, ny))];
  },
  _hit(ev) {
    const c = $("cornerCanvas"), rect = c.getBoundingClientRect(), r = this._rect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    const HIT = 24;  // generous touch target in display px
    let bestI = -1, bestD = HIT * HIT;
    this.pts.forEach(([x, y], i) => {
      const [dx, dy] = toDisplay(x, y, r);
      const d = (dx - px) ** 2 + (dy - py) ** 2;
      if (d < bestD) { bestD = d; bestI = i; }
    });
    return bestI;
  },
  pointerDown(ev) {
    if (!this.active) return;
    ev.preventDefault();
    $("cornerCanvas").setPointerCapture(ev.pointerId);
    const hit = this._hit(ev);
    if (hit >= 0) { this.dragging = hit; return; }
    if (this.pts.length < 4) { this.pts.push(this._toNative(ev)); this.dragging = this.pts.length - 1; this.redraw(); }
  },
  pointerMove(ev) {
    if (!this.active || this.dragging === null) return;
    this.pts[this.dragging] = this._toNative(ev);
    this.redraw();
  },
  pointerUp() { if (this.dragging !== null) { this.dragging = null; this.redraw(); } },

  redraw() {
    if (!this.frozen) return;
    const c = $("cornerCanvas");
    c.width = c.clientWidth; c.height = c.clientHeight;
    const ctx = c.getContext("2d");
    const r = containRect(this.frozen.width, this.frozen.height, c.width, c.height);
    ctx.drawImage(this.frozen, r.offx, r.offy, this.frozen.width * r.scale, this.frozen.height * r.scale);
    ctx.lineWidth = 2; ctx.strokeStyle = "#4f8cff"; ctx.fillStyle = "#4f8cff"; ctx.font = "14px system-ui";
    // Outline through the *ordered* ring so it never self-crosses, whatever tap order.
    const o = this.pts.length === 4 ? orderCorners(this.pts) : null;
    const ring = o ? [o.tl, o.tr, o.br, o.bl] : this.pts;
    ctx.beginPath();
    ring.forEach(([x, y], i) => { const [dx, dy] = toDisplay(x, y, r); i ? ctx.lineTo(dx, dy) : ctx.moveTo(dx, dy); });
    if (ring.length === 4) ctx.closePath();
    ctx.stroke();
    // With all 4 corners, draw the perspective-correct 8x8 grid to align to the squares.
    if (o) {
      const H = squareToQuad(o.tl, o.tr, o.br, o.bl);
      const seg = (p, q) => {
        const A = toDisplay(p[0], p[1], r), B = toDisplay(q[0], q[1], r);
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      };
      for (let k = 1; k < 8; k++) {
        const t = k / 8;
        const f0 = projectH(H, t, 0), f1 = projectH(H, t, 1);
        const r0 = projectH(H, 0, t), r1 = projectH(H, 1, t);
        ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,0.35)"; seg(f0, f1); seg(r0, r1);
        ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(79,140,255,0.85)"; seg(f0, f1); seg(r0, r1);
      }
      ctx.lineWidth = 2; ctx.strokeStyle = "#4f8cff";
    }
    this.pts.forEach(([x, y], i) => {
      const [dx, dy] = toDisplay(x, y, r);
      ctx.beginPath(); ctx.arc(dx, dy, 8, 0, 7); ctx.fill();
      ctx.fillText(String(i + 1), dx + 11, dy - 11);
    });
    $("cornerHint").textContent = this.pts.length < 4
      ? `Tap the 4 board corners in any order (${this.pts.length}/4). Drag any marker to adjust.`
      : "All 4 placed. Drag any corner until the grid lines up with the board squares, then Save.";
    $("saveCorner").disabled = this.pts.length !== 4;
  },

  async predict() {
    if (!this.frozen || !this.predictAvailable) return;
    $("predictCorner").disabled = true;
    $("cornerHint").textContent = "Predicting corners…";
    try {
      const blob = await new Promise((res) => this.frozen.toBlob(res, "image/jpeg", 0.92));
      const body = new FormData();
      body.append("image", blob, "frame.jpg");
      const { corners: c } = await fetch("/api/corners/predict", { method: "POST", body }).then(j);
      this.pts = [c.top_left, c.top_right, c.bottom_right, c.bottom_left];  // native px; nudge to fix
      this.redraw();
    } catch (e) { toast("Predict failed: " + e.message); }
    finally { $("predictCorner").disabled = false; }
  },

  async save() {
    if (this.pts.length !== 4 || !this.opts.onSave) return;
    await this.opts.onSave(orderCorners(this.pts), this.pts.slice());
  },
};

// Wire the shared modal once the DOM is ready (elements live in every page's markup).
function cvWireCornerModal() {
  const canvas = $("cornerCanvas");
  if (!canvas) return;
  canvas.addEventListener("pointerdown", (e) => CM.pointerDown(e));
  canvas.addEventListener("pointermove", (e) => CM.pointerMove(e));
  canvas.addEventListener("pointerup", () => CM.pointerUp());
  canvas.addEventListener("pointercancel", () => CM.pointerUp());
  $("cornerModal").addEventListener("contextmenu", (e) => e.preventDefault());
  if ($("predictCorner")) $("predictCorner").onclick = () => CM.predict();
  if ($("cancelCorner")) $("cancelCorner").onclick = () => CM.close();
  if ($("saveCorner")) $("saveCorner").onclick = () => CM.save().catch((e) => toast(e.message));
  window.addEventListener("resize", () => { if (CM.active && CM.frozen) CM.redraw(); });
  // Esc/Enter while marking; stopImmediatePropagation so a page's keydown (snap/nav)
  // doesn't also fire on the same key.
  document.addEventListener("keydown", (e) => {
    if (!CM.active) return;
    if (e.code === "Escape") { e.stopImmediatePropagation(); CM.close(); }
    else if (e.code === "Enter" && CM.pts.length === 4) { e.preventDefault(); e.stopImmediatePropagation(); CM.save().catch((err) => toast(err.message)); }
  }, true);
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", cvWireCornerModal);
else cvWireCornerModal();
