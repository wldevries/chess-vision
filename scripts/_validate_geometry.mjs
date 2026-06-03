// Check web/geometry.js reproduces the Python reference (web/_geom_ref.json).
// Run: node scripts/_validate_geometry.mjs
import { readFileSync } from "node:fs";
import { homographyFromLattice, squareForPoint } from "../web/geometry.js";

const ref = JSON.parse(readFileSync(new URL("../web/_geom_ref.json", import.meta.url)));
const Hjs = homographyFromLattice(ref.points81);

let maxHErr = 0;
for (let i = 0; i < 9; i++) maxHErr = Math.max(maxHErr, Math.abs(Hjs[i] - ref.H_py[i]));

// Compare reprojection on the probes (H differs only up to numeric noise; squares must match).
let squareOk = 0, squareTot = 0;
for (const p of ref.probes) {
  const sq = squareForPoint(Hjs, p.pt);
  const want = p.square_py; // null serialized as null
  squareTot++;
  if (sq === want) squareOk++;
  else console.log(`  MISMATCH pt=${p.pt} js=${sq} py=${want} (expect ${p.expect})`);
}

console.log(`H max abs elementwise err vs Python: ${maxHErr.toExponential(3)}`);
console.log(`square_for_point agreement: ${squareOk}/${squareTot}`);
process.exit(maxHErr < 1e-3 && squareOk === squareTot ? 0 : 1);
