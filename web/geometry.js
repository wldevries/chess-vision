// Board geometry: 81 lattice points -> homography -> square assignment.
// A faithful JS port of chessvision/geometry.py + corner_regressor.homography_from_lattice.
// Pure, deterministic math (no model) -- the "deterministic geometry" half of the pipeline,
// kept out of the model on purpose (see project doctrine). Validated numerically against the
// Python (scripts/validate_geometry_js.py).
//
// Canonical board = unit square [0,1]^2 over the 8x8 playing area:
//   u = file axis (0 -> a-side, 1 -> h-side), v = rank axis (0 -> rank 8 / top, 1 -> rank 1).
// Anchors: a8->(0,0), h8->(1,0), a1->(0,1), h1->(1,1).

export const FILES = "abcdefgh";

// 81 canonical lattice points, row-major (j outer, i inner) -- EXACTLY the order the corner
// model emits (chessvision/data/corners.py LATTICE_CANONICAL = [[i/8,j/8] for j in 0..8 for i in 0..8]).
export function latticeCanonical() {
  const pts = [];
  for (let j = 0; j < 9; j++) for (let i = 0; i < 9; i++) pts.push([i / 8, j / 8]);
  return pts;
}

// --- 3x3 matrix helpers (row-major number[9]) ---------------------------------

export function matMul3(a, b) {
  const c = new Array(9).fill(0);
  for (let r = 0; r < 3; r++)
    for (let col = 0; col < 3; col++)
      for (let k = 0; k < 3; k++) c[r * 3 + col] += a[r * 3 + k] * b[k * 3 + col];
  return c;
}

export function matInv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) throw new Error("singular 3x3 matrix");
  const inv = 1 / det;
  return [
    A * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
    B * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
    C * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
  ];
}

// Apply a perspective transform (3x3) to a point [x,y] -> [x',y'].
export function applyH(m, [x, y]) {
  const w = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

// --- symmetric eigensolver (cyclic Jacobi) -----------------------------------
// For an n x n symmetric matrix, returns {values, vectors} with vectors[k] the
// eigenvector (length n) for values[k]. Used to find the null-vector of A^T A
// (smallest eigenvalue) -- the homogeneous DLT solution, in place of an SVD.
function jacobiEigen(Ain, n, sweeps = 100) {
  const A = Ain.map((r) => r.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - sn * akq;
          A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk;
          A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq;
          V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const values = A.map((_, i) => A[i][i]);
  const vectors = values.map((_, j) => V.map((row) => row[j]));
  return { values, vectors };
}

// Hartley normalization: centroid -> origin, mean distance -> sqrt(2). Returns
// {pts: normalized, T: 3x3 conditioning matrix}. Matches corner_regressor._normalize_pts.
function hartley(pts) {
  const n = pts.length;
  let cx = 0, cy = 0;
  for (const [x, y] of pts) { cx += x; cy += y; }
  cx /= n; cy /= n;
  let d = 0;
  for (const [x, y] of pts) d += Math.hypot(x - cx, y - cy);
  d /= n;
  const s = Math.SQRT2 / (d + 1e-12);
  const out = pts.map(([x, y]) => [(x - cx) * s, (y - cy) * s]);
  const T = [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
  return { pts: out, T };
}

// Fit canonical->image homography from N correspondences via a normalized DLT
// (unweighted -- confidence weighting is a documented wash). Mirrors
// corner_regressor.homography_from_lattice. src/dst are arrays of [x,y].
export function homographyFromCorrespondences(src, dst) {
  const { pts: sn, T: Ts } = hartley(src);
  const { pts: dn, T: Td } = hartley(dst);
  // Build A^T A (9x9) from the 2N DLT rows without materializing all rows.
  const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const addRow = (row) => {
    for (let a = 0; a < 9; a++) for (let b = 0; b < 9; b++) AtA[a][b] += row[a] * row[b];
  };
  for (let k = 0; k < sn.length; k++) {
    const [x, y] = sn[k], [u, v] = dn[k];
    addRow([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    addRow([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  const { values, vectors } = jacobiEigen(AtA, 9);
  let mi = 0;
  for (let i = 1; i < 9; i++) if (values[i] < values[mi]) mi = i;
  const h = vectors[mi]; // smallest-eigenvalue eigenvector = null space of A
  // Denormalize: H = inv(Td) @ Hn @ Ts, then scale so H[8] == 1.
  let H = matMul3(matMul3(matInv3(Td), h), Ts);
  H = H.map((x) => x / H[8]);
  return H;
}

export function homographyFromLattice(points81) {
  return homographyFromCorrespondences(latticeCanonical(), points81);
}

// Project the 9x9 lattice through H into image pixels, as a 9x9 grid of [x,y] (for drawing
// the board grid overlay). lattice[j][i] = applyH(H, [i/8, j/8]).
export function projectLattice(H) {
  const grid = [];
  for (let j = 0; j < 9; j++) {
    const row = [];
    for (let i = 0; i < 9; i++) row.push(applyH(H, [i / 8, j / 8]));
    grid.push(row);
  }
  return grid;
}

// --- square assignment --------------------------------------------------------

// Canonical (u,v) -> algebraic square, or null if off-board. tol widens the
// on-board test (canonical units; one square = 1/8) so a base just past the far edge resolves.
export function uvToSquare(u, v, tol = 0.0) {
  if (u < -tol || u > 1 + tol || v < -tol || v > 1 + tol) return null;
  const fileIdx = Math.min(Math.max(Math.floor(u * 8), 0), 7);
  const rank = 8 - Math.min(Math.max(Math.floor(v * 8), 0), 7);
  return `${FILES[fileIdx]}${rank}`;
}

// Map an image point (a piece's contact point) to its square via inv(H). null if off-board.
export function squareForPoint(H, pt, tol = 0.0) {
  const [u, v] = applyH(matInv3(H), pt);
  return uvToSquare(u, v, tol);
}

// Rotate an 8x8 board (object {square: value}) by k*90deg clockwise, for the 4-way
// orientation toggle (which physical corner is a8 is not geometry-recoverable -- the
// user picks). k in 0..3. Returns a new {square: value}.
export function rotateBoard(board, k) {
  k = ((k % 4) + 4) % 4;
  if (k === 0) return { ...board };
  const out = {};
  for (const [sq, val] of Object.entries(board)) {
    let f = FILES.indexOf(sq[0]); // 0..7
    let r = parseInt(sq[1], 10) - 1; // 0..7
    for (let s = 0; s < k; s++) {
      const nf = r, nr = 7 - f; // 90deg clockwise
      f = nf; r = nr;
    }
    out[`${FILES[f]}${r + 1}`] = val;
  }
  return out;
}
