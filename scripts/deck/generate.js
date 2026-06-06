// SATA progress-report deck (v0) — built with pptxgenjs per the pptx skill.
// Palette: bio-inspired + control. Motif: number chip + left accent bar.

const pptxgen = require("pptxgenjs");
const path = require("path");

// ---- palette --------------------------------------------------------------
const DARK = "12302D";   // deep pine-teal  (header bands, conclusion)
const TEAL = "0E7C7B";   // primary
const TEALL = "DCEDEA";  // light teal fill
const AMBER = "E0922B";  // sharp accent (torque / energy)
const AMBERD = "9A5F12"; // amber text-safe on light
const CLAY = "B0573F";   // position-control (rigid / negative)
const CLAYL = "F1E2DD";  // light clay fill
const CREAM = "F4F1EA";  // content background
const INK = "243230";    // body text
const MUTE = "7C8A87";   // captions
const WHITE = "FFFFFF";
const LINEC = "E5DFD4";  // card border
const ICE = "BFE0DC";    // light teal text on dark
const FAINT = "1C403C";  // faint motif fill on dark
const FAINTL = "2A534F"; // faint motif line on dark
const FAINTT = "3C6F6B"; // faint motif text on dark

const HEAD = "Georgia";
const BODY = "Calibri";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.33, height: 7.5 });
pres.layout = "W";
pres.author = "SATA Progress Report";
pres.title = "Bio-inspired Adaptive Locomotion via Torque-based Learning";

const sh = () => ({ type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.16 });

function chip(s, x, y, d, label, fill, tc = WHITE, fs = 20) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(String(label), { x, y, w: d, h: d, align: "center", valign: "middle",
    fontSize: fs, bold: true, color: tc, fontFace: HEAD, margin: 0 });
}

function card(s, x, y, w, h, accent) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE },
    line: { color: LINEC, width: 1 }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.1, h, fill: { color: accent } });
}

function header(s, n, title, fs = 25) {
  s.background = { color: CREAM };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 1.05, fill: { color: DARK } });
  chip(s, 0.5, 0.27, 0.52, n, AMBER, DARK, 22);
  s.addText(title, { x: 1.22, y: 0.16, w: 11.7, h: 0.72, fontSize: fs, bold: true,
    color: WHITE, fontFace: HEAD, valign: "middle", margin: 0 });
}

function caption(s, y, txt) {
  s.addText(txt, { x: 0.55, y, w: 12.2, h: 0.4, fontSize: 13, italic: true,
    color: MUTE, fontFace: BODY, margin: 0 });
}

function refs(s, txt) {
  s.addText([{ text: "Refs  ", options: { bold: true, color: TEAL } },
             { text: txt, options: { color: MUTE } }],
    { x: 0.55, y: 7.06, w: 12.2, h: 0.32, fontSize: 10, italic: true, fontFace: BODY, margin: 0 });
}

function down(s, x, y, color) {
  s.addText("▼", { x, y, w: 0.4, h: 0.28, align: "center", valign: "middle",
    fontSize: 12, color, margin: 0 });
}

// ---- icon rendering (react-icons -> PNG) ----------------------------------
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const FA = require("react-icons/fa");
const TB = require("react-icons/tb");
async function icoPng(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
function img(s, data, x, y, w, h) { s.addImage({ data, x, y, w, h }); }

(async () => {
const IC = {};
const specs = {
  dogTeal: [FA.FaDog, TEAL], dogClay: [FA.FaDog, CLAY], dogInk: [FA.FaDog, "566460"],
  arrowR: [FA.FaArrowRight, MUTE], forceDown: [FA.FaArrowDown, CLAY],
  ok: [FA.FaCheckCircle, TEAL], no: [FA.FaTimesCircle, CLAY],
  wave: [TB.TbWaveSine, TEAL], dumb: [FA.FaDumbbell, TEAL], batt: [FA.FaBatteryQuarter, AMBERD],
};
await Promise.all(Object.entries(specs).map(async ([k, [C, col]]) => { IC[k] = await icoPng(C, col); }));

// ============================================================ TITLE
let s = pres.addSlide();
s.background = { color: DARK };

// faint torque-control flow motif (right side, low contrast)
const mX = 9.55, mW = 2.95;
const motif = ["Command", "Torque", "Motion"];
let mY = 1.7;
for (let i = 0; i < motif.length; i++) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: mX, y: mY, w: mW, h: 0.8, fill: { color: FAINT }, line: { color: FAINTL, width: 1 }, rectRadius: 0.08 });
  s.addText(motif[i], { x: mX, y: mY, w: mW, h: 0.8, align: "center", valign: "middle", fontSize: 14, color: FAINTT, fontFace: BODY, margin: 0 });
  if (i < motif.length - 1)
    s.addText("▼", { x: mX + mW / 2 - 0.2, y: mY + 0.82, w: 0.4, h: 0.3, align: "center", valign: "middle", fontSize: 12, color: FAINTL, margin: 0 });
  mY += 1.3;
}

chip(s, 0.7, 0.82, 0.62, "τ", AMBER, DARK, 24);
s.addText("PROGRESS REPORT   ·   ADAPTIVE CONTROL SYSTEMS",
  { x: 1.5, y: 0.9, w: 9.0, h: 0.45, fontSize: 12.5, bold: true, color: AMBER, fontFace: BODY, charSpacing: 1, valign: "middle", margin: 0 });

s.addText("Bio-inspired Adaptive Locomotion\nvia Torque-based Learning",
  { x: 0.7, y: 2.5, w: 8.7, h: 1.4, fontSize: 31, bold: true, color: WHITE, fontFace: HEAD, lineSpacingMultiple: 1.08, margin: 0, valign: "top" });
s.addText("A Case Study of SATA",
  { x: 0.72, y: 3.78, w: 8.4, h: 0.45, fontSize: 20, italic: true, color: ICE, fontFace: HEAD, margin: 0 });
s.addText("Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning",
  { x: 0.72, y: 4.24, w: 8.5, h: 0.4, fontSize: 12, italic: true, color: "8FA6A3", fontFace: BODY, margin: 0 });

s.addShape(pres.shapes.RECTANGLE, { x: 0.74, y: 4.85, w: 2.0, h: 0.05, fill: { color: AMBER } });
s.addText("Chuan-Han Li", { x: 0.7, y: 5.08, w: 8, h: 0.45, fontSize: 17, bold: true, color: WHITE, fontFace: BODY, margin: 0 });
s.addText("2026 / 05 / 25", { x: 0.7, y: 5.54, w: 8, h: 0.4, fontSize: 14, color: MUTE, fontFace: BODY, margin: 0 });
s.addText([{ text: "Main reference:  ", options: { bold: true, color: AMBER } },
           { text: "Li et al., SATA (RSS 2025)", options: { color: MUTE } }],
  { x: 0.7, y: 6.96, w: 9.0, h: 0.35, fontSize: 11, italic: true, fontFace: BODY, margin: 0 });
s.addNotes(
  "[~10 sec]\n"
  + "Open: title, course, your name — one breath, then move on.\n"
  + "CORE (set the frame early): We are NOT proposing a new controller. We reproduce SATA, "
  + "understand its adaptive behavior, and discuss how it answers adaptive-control questions.\n"
  + "TRANSITION: 'Let me start with why this is a control problem.'");

// ============================================================ SLIDE 1
s = pres.addSlide();
header(s, 1, "Locomotion Is a Control Problem");

// LEFT — problem card
card(s, 0.55, 1.45, 5.45, 1.5, CLAY);
s.addText("THE PROBLEM", { x: 0.8, y: 1.58, w: 5.0, h: 0.32, fontSize: 13, bold: true,
  color: CLAY, fontFace: BODY, charSpacing: 0.5, margin: 0 });
s.addText("Position-based locomotion is accurate under known conditions but rigid — "
  + "it struggles with compliance, disturbances, unknown terrain, and the sim-to-real gap.",
  { x: 0.8, y: 1.92, w: 5.0, h: 0.95, fontSize: 14, color: INK, fontFace: BODY, lineSpacingMultiple: 1.08, margin: 0, valign: "top" });

// situational scene: rigid control struggles on unknown / soft terrain
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.55, y: 3.05, w: 5.45, h: 1.36, fill: { color: "EFEBE3" }, line: { color: LINEC, width: 1 }, rectRadius: 0.06 });
s.addText("RIGID CONTROL ON UNKNOWN TERRAIN", { x: 0.72, y: 3.12, w: 5.1, h: 0.26, fontSize: 10, bold: true, color: CLAY, fontFace: BODY, charSpacing: 0.5, margin: 0 });
// left: firm / known (solid ground)
img(s, IC.dogTeal, 1.2, 3.46, 0.56, 0.56);
s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.04, w: 1.55, h: 0, line: { color: "9A8C7A", width: 3 } });
s.addText("firm · known", { x: 0.7, y: 4.18, w: 1.75, h: 0.22, fontSize: 10, color: MUTE, fontFace: BODY, align: "center", margin: 0 });
// arrow
img(s, IC.arrowR, 2.62, 3.66, 0.5, 0.34);
// right: soft / unknown (loose dashed ground that sags where the leg steps) + excessive force
img(s, IC.dogClay, 3.95, 3.54, 0.56, 0.56);
img(s, IC.forceDown, 4.64, 3.32, 0.28, 0.4);
img(s, IC.no, 5.52, 3.5, 0.24, 0.24);
s.addShape(pres.shapes.LINE, { x: 3.45, y: 4.04, w: 0.75, h: 0.09, line: { color: CLAY, width: 3, dashType: "dash" } });
s.addShape(pres.shapes.LINE, { x: 4.20, y: 4.13, w: 0.58, h: 0, line: { color: CLAY, width: 3, dashType: "dash" } });
s.addShape(pres.shapes.LINE, { x: 4.78, y: 4.13, w: 0.92, h: -0.09, line: { color: CLAY, width: 3, dashType: "dash" } });
s.addText("soft · unknown", { x: 3.55, y: 4.18, w: 2.05, h: 0.22, fontSize: 10, color: CLAY, fontFace: BODY, align: "center", margin: 0 });

// bridge caption
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.52, w: 0.08, h: 0.62, fill: { color: TEAL } });
s.addText("→  Torque control may improve compliance, providing conditions for more adaptive locomotion.",
  { x: 0.75, y: 4.5, w: 5.25, h: 0.66, fontSize: 12.5, bold: true, color: TEAL, fontFace: BODY, valign: "middle", margin: 0 });

// RIGHT — comparison columns
const colY = 1.45;
const posX = 6.35, torX = 9.85, colW = 2.95;
s.addShape(pres.shapes.RECTANGLE, { x: posX, y: colY, w: colW, h: 0.5, fill: { color: CLAY } });
s.addText("POSITION CONTROL", { x: posX, y: colY, w: colW, h: 0.5, align: "center", valign: "middle",
  fontSize: 12.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 0.5, margin: 0 });
s.addShape(pres.shapes.RECTANGLE, { x: torX, y: colY, w: colW, h: 0.5, fill: { color: TEAL } });
s.addText("TORQUE CONTROL", { x: torX, y: colY, w: colW, h: 0.5, align: "center", valign: "middle",
  fontSize: 12.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 0.5, margin: 0 });

const posSteps = ["Command position", "Rigid behavior", "Poor adaptation"];
const torSteps = ["Command force", "Compliant interaction", "Adaptive response"];
let yy = 2.15;
for (let i = 0; i < 3; i++) {
  s.addShape(pres.shapes.RECTANGLE, { x: posX, y: yy, w: colW, h: 0.6, fill: { color: CLAYL }, line: { color: CLAY, width: 1 } });
  s.addText(posSteps[i], { x: posX, y: yy, w: colW, h: 0.6, align: "center", valign: "middle", fontSize: 13, color: INK, fontFace: BODY, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: torX, y: yy, w: colW, h: 0.6, fill: { color: TEALL }, line: { color: TEAL, width: 1 } });
  s.addText(torSteps[i], { x: torX, y: yy, w: colW, h: 0.6, align: "center", valign: "middle", fontSize: 13, color: INK, fontFace: BODY, margin: 0 });
  if (i < 2) {
    down(s, posX + colW / 2 - 0.2, yy + 0.61, CLAY);
    down(s, torX + colW / 2 - 0.2, yy + 0.61, TEAL);
  }
  yy += 0.95;
}

// BOTTOM — research question
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.5, w: 12.23, h: 1.0, fill: { color: DARK } });
s.addText("RESEARCH QUESTION", { x: 0.85, y: 5.62, w: 11.6, h: 0.3, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 0.5, margin: 0 });
s.addText("How can robots achieve safer and more adaptive locomotion in unknown environments?",
  { x: 0.85, y: 5.92, w: 11.6, h: 0.5, fontSize: 16.5, bold: true, color: WHITE, fontFace: HEAD, margin: 0 });
s.addText([{ text: "Adaptive behavior ", options: {} },
           { text: "≠", options: { bold: true, color: CLAY } },
           { text: " adaptive control", options: {} }],
  { x: 6.3, y: 4.92, w: 6.5, h: 0.4, fontSize: 13, italic: true, bold: true, color: INK, fontFace: BODY, align: "right", margin: 0 });
refs(s, "SATA · Chen et al. (torque control) · Lee et al. (terrain) · Miki et al. (perceptive loco.)");
s.addNotes(
  "[~40 sec]\n"
  + "CORE MESSAGE (repeat across the talk): We are NOT proposing a new controller. "
  + "We aim to reproduce SATA, understand its adaptive behavior, and discuss how it answers adaptive-control questions.\n"
  + "TALKING POINTS:\n"
  + "- Frame this as a CONTROL problem, not an RL talk.\n"
  + "- Position control commands a joint angle; a low-level PD loop turns the error into torque. Stiff and accurate when the world is known.\n"
  + "- It fails on exactly what this course cares about: compliance, disturbance rejection, unknown terrain, sim-to-real gap.\n"
  + "- Torque control commands force directly, so the robot can yield and adapt instead of resisting.\n"
  + "TRANSITION: 'The question is how to get adaptive locomotion in unknown environments — SATA is one bio-inspired answer.'");

// ============================================================ SLIDE 2
s = pres.addSlide();
header(s, 2, "SATA Framework: Bio-inspired Adaptation in Torque Control", 22);
caption(s, 1.18, "An RL torque policy wrapped by a biomechanical layer and a growth curriculum.");

// LEFT — vertical flow
const fx = 2.75, fw = 3.55;
const nodes = ["Observation", "Torque Policy (RL)", "Biomechanical Layer", "Torque Output", "Robot"];
const hot = [false, true, true, false, false];
let ny = 1.75;
const centers = [];
for (let i = 0; i < nodes.length; i++) {
  const fill = hot[i] ? TEALL : WHITE;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: ny, w: fw, h: 0.6, fill: { color: fill }, line: { color: hot[i] ? TEAL : LINEC, width: hot[i] ? 1.5 : 1 }, shadow: sh() });
  s.addText(nodes[i], { x: fx, y: ny, w: fw, h: 0.6, align: "center", valign: "middle", fontSize: 14, bold: hot[i], color: INK, fontFace: BODY, margin: 0 });
  centers.push(ny + 0.3);
  if (i < nodes.length - 1) down(s, fx + fw / 2 - 0.2, ny + 0.62, MUTE);
  ny += 0.92;
}
// feedback loop (Robot -> rail -> back into Torque Policy)
const rail = 2.4;
s.addShape(pres.shapes.LINE, { x: rail, y: centers[4], w: fx - rail, h: 0, line: { color: AMBER, width: 2.5 } });        // robot -> rail
s.addShape(pres.shapes.LINE, { x: rail, y: centers[1], w: 0, h: centers[4] - centers[1], line: { color: AMBER, width: 2.5 } }); // rail up
s.addShape(pres.shapes.LINE, { x: rail, y: centers[1], w: fx - rail, h: 0, line: { color: AMBER, width: 2.5, endArrowType: "triangle" } }); // rail -> into policy
s.addText("State feedback\n(base + fatigue)", { x: 0.4, y: (centers[1] + centers[4]) / 2 - 0.45, w: 1.85, h: 0.9,
  align: "right", valign: "middle", fontSize: 12, bold: true, color: AMBERD, fontFace: BODY, margin: 0 });
img(s, IC.dogInk, fx + 0.5, centers[4] - 0.19, 0.38, 0.38);  // Go2 in the Robot node

// RIGHT — highlight cards
const hx = 7.05, hw = 5.75;
card(s, hx, 1.7, hw, 1.6, TEAL);
chip(s, hx + 0.28, 1.92, 0.55, "M", TEAL, WHITE, 18);
s.addText("Biomechanical Model", { x: hx + 1.0, y: 1.9, w: hw - 1.2, h: 0.4, fontSize: 18, bold: true, color: TEAL, fontFace: HEAD, margin: 0 });
const micons = [[IC.wave, "activation"], [IC.dumb, "muscle"], [IC.batt, "fatigue"]];
let mix = hx + 1.0;
for (const [d, lab] of micons) {
  img(s, d, mix, 2.55, 0.32, 0.32);
  s.addText(lab, { x: mix + 0.38, y: 2.55, w: 1.15, h: 0.32, fontSize: 11.5, color: INK, fontFace: BODY, valign: "middle", margin: 0 });
  mix += 1.55;
}

card(s, hx, 3.45, hw, 1.5, AMBER);
chip(s, hx + 0.28, 3.7, 0.55, "G", AMBER, WHITE, 18);
s.addText("Growth Mechanism", { x: hx + 1.0, y: 3.67, w: hw - 1.2, h: 0.4, fontSize: 18, bold: true, color: AMBERD, fontFace: HEAD, margin: 0 });
s.addText("torque limit  ·  training curriculum  ·  control frequency", { x: hx + 1.0, y: 4.12, w: hw - 1.2, h: 0.7, fontSize: 14, color: INK, fontFace: BODY, lineSpacingMultiple: 1.08, margin: 0, valign: "top" });

s.addShape(pres.shapes.RECTANGLE, { x: hx, y: 5.15, w: hw, h: 0.95, fill: { color: "ECF1F0" }, line: { color: LINEC, width: 1 } });
s.addText([{ text: "Goal:  ", options: { bold: true, color: TEAL } },
           { text: "smoother torque generation and improved robustness through bio-inspired adaptation.", options: {} }],
  { x: hx + 0.18, y: 5.15, w: hw - 0.36, h: 0.95, fontSize: 13, color: INK, fontFace: BODY, lineSpacingMultiple: 1.08, margin: 0, valign: "middle" });
refs(s, "SATA · Hill (muscle dynamics) · Liu et al. (fatigue) · Bellegarda & Ijspeert (CPG-RL)");
s.addNotes(
  "[~55 sec]\nTALKING POINTS:\n"
  + "- One message: SATA injects bio-inspired adaptation INTO a torque policy. Don't read equations.\n"
  + "- Flow: observation -> RL torque policy -> biomechanical layer -> torque -> robot; fatigue state feeds back.\n"
  + "- Biomechanical model: activation smooths the command, a muscle model prevents abrupt torque, fatigue discourages overusing joints.\n"
  + "- Growth mechanism: torque limit, reward, and control frequency unlock progressively — like an animal maturing.\n"
  + "- Net effect: smoother, more robust torques plus generalization, demonstrated zero-shot.\n"
  + "TRANSITION: 'That's what SATA is — here is how WE plan to study it.'");

// ============================================================ SLIDE 3
s = pres.addSlide();
header(s, 3, "Plan: Understand → Reproduce → Analyze");
caption(s, 1.18, "A feasible 3-phase study, with an optional extension — we study SATA, not redesign the RL.");

const phases = [
  { n: "1", t: "Reproduce", c: TEAL, b: "Set up Isaac Gym on a CUDA server (container / venv) and run the official SATA training.", out: "simulation running", opt: false },
  { n: "2", t: "Ablation", c: TEAL, b: "Disable fatigue · change torque limit · modify growth schedule · vary terrain.", out: "behavioral data", opt: false },
  { n: "3", t: "Control Perspective", c: TEAL, b: "Questions from adaptive control:\nGrowth ↔ gain scheduling\nFatigue ↔ feedback\nTorque modulation ↔ robustness", out: "comparison & discussion", opt: false },
  { n: "4", t: "Residual Compensation", c: "8E9B98", out: "preliminary observations", opt: true,
    rich: [
      { text: "Preliminary exploration.", options: { breakLine: true } },
      { text: " ", options: { breakLine: true } },
      { text: "τ", options: {} }, { text: "total", options: { subscript: true } },
      { text: " = τ", options: {} }, { text: "SATA", options: { subscript: true } },
      { text: " + τ", options: {} }, { text: "comp", options: { subscript: true, breakLine: true } },
      { text: " ", options: { breakLine: true } },
      { text: "Only if time & feasibility allow.", options: {} },
    ] },
];
const pTop = 1.7, pH = 3.15, pW = 2.86, pGap = 0.3;
let px = 0.55;
for (let i = 0; i < phases.length; i++) {
  const p = phases[i];
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: pTop, w: pW, h: pH, fill: { color: WHITE }, line: { color: LINEC, width: 1 }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: pTop, w: pW, h: 0.85, fill: { color: p.c } });
  chip(s, px + 0.18, pTop + 0.17, 0.5, p.n, WHITE, p.c, 18);
  s.addText(p.opt ? "PHASE 4 · OPTIONAL" : "PHASE " + p.n, { x: px + 0.8, y: pTop + 0.13, w: pW - 0.9, h: 0.28, fontSize: 9.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 0.5, margin: 0 });
  s.addText(p.t, { x: px + 0.8, y: pTop + 0.4, w: pW - 0.9, h: 0.4, fontSize: 14, bold: true, color: WHITE, fontFace: BODY, margin: 0, valign: "top" });
  s.addText(p.rich || p.b, { x: px + 0.22, y: pTop + 1.0, w: pW - 0.44, h: 1.4, fontSize: 11.5, color: INK, fontFace: BODY, lineSpacingMultiple: 1.12, margin: 0, valign: "top" });
  // output footer inside card
  s.addShape(pres.shapes.LINE, { x: px + 0.22, y: pTop + pH - 0.72, w: pW - 0.44, h: 0, line: { color: LINEC, width: 1 } });
  s.addText([{ text: "OUTPUT  ", options: { bold: true, color: p.opt ? MUTE : TEAL } },
             { text: p.out, options: { color: INK } }],
    { x: px + 0.22, y: pTop + pH - 0.62, w: pW - 0.44, h: 0.5, fontSize: 11.5, fontFace: BODY, margin: 0, valign: "top" });
  if (i < phases.length - 1)
    s.addText("→", { x: px + pW - 0.02, y: pTop + 0.2, w: pGap + 0.04, h: 0.5, align: "center", valign: "middle", fontSize: 18, bold: true, color: MUTE, margin: 0 });
  px += pW + pGap;
}

s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.55, w: 12.23, h: 0.95, fill: { color: DARK } });
s.addText("FRAMING", { x: 0.85, y: 5.66, w: 11.6, h: 0.3, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 0.5, margin: 0 });
s.addText("Adaptive control supplies the questions; SATA offers a different kind of answer.",
  { x: 0.85, y: 5.94, w: 11.6, h: 0.45, fontSize: 15.5, bold: true, color: WHITE, fontFace: HEAD, margin: 0 });
refs(s, "SATA · RL2AC · DecAP");
s.addNotes(
  "[~55 sec]\nTALKING POINTS:\n"
  + "- Feasibility is the point of this slide: executable, not just a paper review.\n"
  + "- Phase 1: stand up the official SATA pipeline in Isaac Gym — proves we can run it.\n"
  + "- Phase 2: controlled ablations — toggle fatigue, retune torque limit / growth, change terrain; observe behavior shifts.\n"
  + "- Phase 3: compare each mechanism with the adaptive-control question it echoes (analogy, not equivalence).\n"
  + "- Phase 4 residual term only if time allows — not promised.\n"
  + "TRANSITION: 'So what do we expect to deliver?'");

// ============================================================ SLIDE 4
s = pres.addSlide();
header(s, 4, "Expected Outputs & Contribution");

s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 12.23, h: 0.8, fill: { color: TEALL } });
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 0.1, h: 0.8, fill: { color: TEAL } });
s.addText("We study adaptive behavior — not propose a new RL algorithm.",
  { x: 0.85, y: 1.4, w: 11.8, h: 0.8, fontSize: 18, bold: true, color: TEAL, fontFace: HEAD, valign: "middle", margin: 0 });

// LEFT — expected outputs
s.addText("EXPECTED OUTPUTS", { x: 0.55, y: 2.55, w: 5.9, h: 0.4, fontSize: 15, bold: true, color: DARK, fontFace: BODY, charSpacing: 0.5, margin: 0 });
const outs = ["Simulation reproduction", "Behavior analysis (ablation)",
              "Control-perspective comparison", "Literature-grounded discussion"];
let oy = 3.05;
for (let i = 0; i < outs.length; i++) {
  chip(s, 0.6, oy, 0.42, i + 1, TEAL, WHITE, 14);
  s.addText(outs[i], { x: 1.18, y: oy, w: 5.3, h: 0.42, fontSize: 15, color: INK, fontFace: BODY, valign: "middle", margin: 0 });
  oy += 0.74;
}

// RIGHT — research questions (2x2)
s.addText("RESEARCH QUESTIONS", { x: 6.9, y: 2.55, w: 5.9, h: 0.4, fontSize: 15, bold: true, color: DARK, fontFace: BODY, charSpacing: 0.5, margin: 0 });
const rqs = [["RQ1", "Why torque control?"], ["RQ2", "What creates adaptation?"],
             ["RQ3", "How does SATA address problems traditionally studied in adaptive control?"], ["RQ4", "Can lightweight compensation help?"]];
const rqW = 2.86, rqH = 1.3, rqGap = 0.2;
for (let i = 0; i < 4; i++) {
  const cx = 6.9 + (i % 2) * (rqW + rqGap);
  const cy = 3.1 + Math.floor(i / 2) * (rqH + rqGap);
  card(s, cx, cy, rqW, rqH, i === 3 ? AMBER : TEAL);
  s.addText(rqs[i][0], { x: cx + 0.22, y: cy + 0.15, w: rqW - 0.4, h: 0.35, fontSize: 13, bold: true, color: i === 3 ? AMBERD : TEAL, fontFace: HEAD, margin: 0 });
  s.addText(rqs[i][1], { x: cx + 0.22, y: cy + 0.46, w: rqW - 0.4, h: rqH - 0.58, fontSize: 12, color: INK, fontFace: BODY, lineSpacingMultiple: 1.1, margin: 0, valign: "top" });
}

// BOTTOM — conclusion
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 6.15, w: 12.23, h: 1.0, fill: { color: DARK } });
s.addText("IN ONE LINE", { x: 0.85, y: 6.26, w: 11.6, h: 0.28, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 0.5, margin: 0 });
s.addText("This project studies how bio-inspired torque control produces adaptive locomotion, and asks how SATA answers questions traditionally posed by adaptive and robust control.",
  { x: 0.85, y: 6.52, w: 11.6, h: 0.55, fontSize: 14, bold: true, color: WHITE, fontFace: BODY, margin: 0, valign: "top" });
s.addNotes(
  "[~30 sec]\nTALKING POINTS:\n"
  + "- Close the loop, no overclaiming: the contribution is understanding, not a new algorithm.\n"
  + "- Three concrete outputs: reproduction, behavior analysis, control-perspective comparison.\n"
  + "- The four research questions map onto the four slides.\n"
  + "CLOSING LINE (read the bottom band): bio-inspired torque control -> adaptive behavior -> answers questions posed by adaptive & robust control.\n"
  + "REINFORCE THE CORE: we are not proposing a new controller — reproduce SATA, understand its adaptive behavior, discuss how it answers adaptive-control questions.\n"
  + "TRANSITION: 'Happy to take questions.'");

const out = path.resolve(__dirname, "../../docs/20260525_progress_report_v1.pptx");
await pres.writeFile({ fileName: out });
console.log("saved", out);
})();
