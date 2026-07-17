import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const ARTIFACT_TOOL =
  "C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const PYTHON =
  "C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";

const {
  Presentation,
  PresentationFile,
  text,
} = await import(pathToFileURL(ARTIFACT_TOOL).href);

const root = process.cwd();
const outDir = path.join(root, "output");
const scratchDir = path.join(root, "scratch", "flowchart");
const assetDir = path.join(scratchDir, "assets");
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(scratchDir, { recursive: true });

const W = 2816;
const H = 1536;

const pptxPath = path.join(outDir, "流程图.pptx");
const previewPath = path.join(scratchDir, "流程图.preview.png");
const reopenPreviewPath = path.join(scratchDir, "流程图.reopen-preview.png");

const C = {
  black: "#000000",
  gold: "#9B6A00",
  amber: "#FFD979",
  amber2: "#FFF0B8",
  blue: "#2A5E8E",
  blueFill: "#EAF5FF",
  green: "#4F7F27",
  greenFill: "#DFF2CF",
  greenFill2: "#CFE7B8",
  purple: "#6F237D",
  purpleFill: "#EAD6F5",
  grayFill: "#ECECEC",
  orange: "#F5B52E",
  tokenBlue: "#6DA7D9",
  tokenGreen: "#A7CD73",
};

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
  title: "流程图",
});

const slide = presentation.slides.add();

async function saveBlob(blob, filePath) {
  if (typeof blob.save === "function") {
    await blob.save(filePath);
    return;
  }
  if (typeof blob.arrayBuffer === "function") {
    await fs.promises.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
    return;
  }
  if (blob instanceof Uint8Array || Buffer.isBuffer(blob)) {
    await fs.promises.writeFile(filePath, blob);
    return;
  }
  throw new TypeError(`Unsupported blob type for ${filePath}`);
}

function addShape(geometry, x, y, w, h, opts = {}) {
  const sh = slide.shapes.add({ geometry });
  sh.frame = { left: x, top: y, width: w, height: h };
  if (opts.rotate !== undefined) sh.position.rotation = opts.rotate;
  if (opts.fill) sh.fill = opts.fill;
  if (opts.line) sh.line = opts.line;
  if (opts.shadow) sh.shadow = opts.shadow;
  if (opts.radius) sh.borderRadius = opts.radius;
  return sh;
}

function addBox(x, y, w, h, fill, line = C.black, width = 2, radius = false) {
  return addShape(radius ? "roundRect" : "rect", x, y, w, h, {
    fill,
    line: { fill: line, width },
  });
}

function addDashedBox(x, y, w, h, radius = true) {
  return addShape(radius ? "roundRect" : "rect", x, y, w, h, {
    line: { fill: C.black, width: 4, style: "dash" },
  });
}

function addText(value, x, y, w, h, style = {}) {
  const base = {
    typeface: "Times New Roman",
    fontSize: style.fontSize ?? 32,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    color: style.color ?? C.black,
    alignment: style.alignment ?? "center",
    insets: style.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  slide.compose(
    text(value, {
      width: w,
      height: h,
      style: base,
    }),
    { frame: { left: x, top: y, width: w, height: h }, baseUnit: 1 },
  );
}

function addLine(x1, y1, x2, y2, color = C.black, width = 4, dashed = false) {
  return addShape("line", Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1), {
    line: { fill: color, width, ...(dashed ? { style: "dash" } : {}) },
  });
}

function arrowHead(x, y, direction = "right", size = 26, color = C.black) {
  const rotation = { up: 0, right: 90, down: 180, left: 270 }[direction] ?? 90;
  let left = x - size / 2;
  let top = y - size / 2;
  if (direction === "right") left = x - size;
  if (direction === "left") left = x;
  if (direction === "down") top = y - size;
  if (direction === "up") top = y;
  return addShape("triangle", left, top, size, size, {
    fill: color,
    rotate: rotation,
  });
}

function arrow(x1, y1, x2, y2, color = C.black, width = 4, size = 26) {
  addLine(x1, y1, x2, y2, color, width);
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dir = Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? "right" : "left") : (dy >= 0 ? "down" : "up");
  arrowHead(x2, y2, dir, size, color);
}

function polyArrow(points, color = C.black, width = 4, size = 26) {
  for (let i = 0; i < points.length - 1; i += 1) {
    addLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], color, width);
  }
  const a = points[points.length - 2];
  const b = points[points.length - 1];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dir = Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? "right" : "left") : (dy >= 0 ? "down" : "up");
  arrowHead(b[0], b[1], dir, size, color);
}

function image(name, x, y, w, h, alt) {
  const p = path.join(assetDir, name);
  slide.images.add({ path: p, frame: { left: x, top: y, width: w, height: h }, fit: "cover", alt });
}

function tokenColumn(x, y, count = 4, fill = C.tokenBlue, scale = 1) {
  for (let i = 0; i < count; i += 1) {
    addBox(x, y + i * 31 * scale, 28 * scale, 28 * scale, fill, "#1C5A86", 2, false);
  }
}

function tokenDots(x, y) {
  for (let i = 0; i < 3; i += 1) addShape("ellipse", x + i * 38, y, 10, 10, { fill: "#6D9AC8" });
}

function visualBlock(x, y, w, h) {
  addShape("parallelogram", x - 18, y + 18, 22, h, { fill: "#FFE2A1", line: { fill: C.gold, width: 3 } });
  addShape("parallelogram", x, y - 18, w, 22, { fill: "#FFF1C8", line: { fill: C.gold, width: 3 } });
  addBox(x, y, w, h, "#F8D890", C.gold, 3, false);
}

function featureStack(x, y, s = 1) {
  const blocks = [
    [0, 0, 58, 230],
    [100, 50, 50, 190],
    [190, 90, 54, 150],
    [280, 132, 66, 104],
    [380, 155, 78, 80],
    [500, 175, 48, 58],
  ];
  for (const [dx, dy, bw, bh] of blocks) visualBlock(x + dx * s, y + dy * s, bw * s, bh * s);
}

function tinyVisualTokens(x, y, n = 3) {
  for (let i = 0; i < n; i += 1) addBox(x + i * 46, y, 30, 60, C.amber, C.gold, 2, true);
}

function label(value, x, y, w, h, size = 34, bold = true) {
  addText(value, x, y, w, h, { fontSize: size, bold, alignment: "center" });
}

// Background
addBox(0, 0, W, H, "#FFFFFF", "#FFFFFF", 0, false).sendToBack();

// Title
addText("Overview of the M²VG Architecture", 620, 10, 1580, 86, {
  fontSize: 56,
  bold: true,
});

// Left top primary input and DFNM
label("Primary\nInput", 75, 155, 165, 90, 36);
image("primary_top.png", 78, 295, 160, 150, "Primary input image crop");
addBox(85, 303, 135, 125, "#F3C750", C.black, 3, false);
addBox(95, 313, 115, 105, "#FFFFFF", C.black, 2, false);
addText("s = (I, T, B, M, c, y)", 0, 492, 315, 48, { fontSize: 33, italic: true, alignment: "left" });

addDashedBox(325, 135, 690, 500, true);
addText("DFNM: Dynamic Failure-case\nNegotiation Module (Only in Training)", 390, 155, 560, 92, {
  fontSize: 39,
  bold: true,
});
addDashedBox(395, 275, 370, 120, true);
addText("Objective Removal\nI → I⁻, T → T, y = 0", 420, 286, 320, 92, { fontSize: 34 });
addDashedBox(395, 485, 420, 120, true);
addText("Semantic Mismatch\nI → I, T → T⁻, y = 0", 420, 498, 370, 90, { fontSize: 34 });
addText("(I⁻, T)", 842, 330, 120, 44, { fontSize: 34, italic: true });
addText("(I, T⁻)", 840, 540, 120, 44, { fontSize: 34, italic: true });
addText("Fallback: Original sample usage or sample if\nreliability conditions aren’t met", 155, 665, 880, 86, {
  fontSize: 35,
});

polyArrow([[237, 425], [350, 425], [350, 335], [395, 335]], C.black, 4, 23);
polyArrow([[237, 425], [350, 425], [350, 545], [395, 545]], C.black, 4, 23);
arrow(765, 335, 835, 335, C.black, 4, 23);
arrow(815, 545, 835, 545, C.black, 4, 23);
polyArrow([[957, 335], [1016, 335], [1016, 445], [1065, 445]], C.black, 4, 23);
polyArrow([[957, 545], [1016, 545], [1016, 445], [1065, 445]], C.black, 4, 23);
addLine(158, 547, 158, 760, C.black, 4, true);
arrowHead(158, 547, "up", 24, C.black);
addLine(158, 760, 445, 760, C.black, 4, true);
polyArrow([[445, 760], [445, 860]], C.black, 4, 22);

// Bottom primary input
addShape("roundRect", 0, 780, 345, 610, {
  fill: C.blueFill,
  line: { fill: C.blue, width: 3 },
});
addText("Primary Input\n(Single Sample)", 25, 805, 275, 86, { fontSize: 35, bold: true });
image("primary_single.png", 76, 915, 150, 155, "Single primary input crop");
addBox(86, 925, 128, 130, "#F3C750", C.black, 3, false);
addBox(98, 937, 104, 106, "#FFFFFF", C.black, 2, false);
addText("Image", 98, 1072, 105, 45, { fontSize: 35 });
addText("Text Expression", 28, 1115, 270, 44, { fontSize: 35 });
addBox(64, 1170, 185, 55, "#E8F2FF", C.blue, 3, true);
addText("attributes", 75, 1174, 160, 46, { fontSize: 31 });
addBox(64, 1240, 185, 55, "#E8F2FF", C.blue, 3, true);
addText("方位", 80, 1244, 150, 46, { fontSize: 31 });
addText("Text", 105, 1305, 100, 45, { fontSize: 35 });
label("Primary Input\n(Single Sample)", 42, 1412, 285, 90, 38);

// Encoder and supervision
arrow(242, 980, 310, 980, C.black, 4, 24);
arrow(260, 1236, 310, 1236, C.black, 4, 24);
addShape("trapezoid", 310, 885, 285, 435, {
  fill: "#FFE9A4",
  line: { fill: C.blue, width: 3 },
});
addText("V&L\nEncoder", 355, 1025, 190, 130, { fontSize: 41, bold: true });
addLine(445, 1320, 445, 1392, C.black, 4);
addLine(445, 1392, 520, 1392, C.black, 4);
addText("Supervision info\n(B, M, c, y)", 415, 1398, 290, 85, { fontSize: 35 });

// Word-level text features
polyArrow([[550, 1040], [585, 1040], [585, 1168], [620, 1168]], C.black, 4, 24);
tokenColumn(620, 1070, 4, C.tokenBlue);
tokenColumn(660, 1070, 4, C.tokenBlue);
tokenDots(730, 1133);
tokenColumn(805, 1070, 4, C.tokenBlue);
addText("Word-level\nText Features\nFₜ", 620, 1230, 250, 130, { fontSize: 34 });

// Lower visual features from encoder
polyArrow([[550, 960], [585, 960], [585, 930], [615, 930]], C.black, 4, 24);
featureStack(625, 890, 0.75);
polyArrow([[800, 900], [800, 815], [1510, 815]], C.gold, 4, 22);
addLine(800, 815, 1168, 815, C.gold, 4);
polyArrow([[1168, 815], [1168, 650]], C.gold, 4, 22);
addText("Compressed into\nvisual tokens", 1185, 660, 280, 100, { fontSize: 32 });

// Top visual branch
addShape("roundRect", 1040, 140, 690, 390, {
  fill: C.amber2,
  line: { fill: C.amber2, width: 1 },
});
addText("Multi-scale Visual Features\n{Fᵛˡ}ˡ₌₁", 1140, 150, 480, 110, { fontSize: 40, bold: true });
featureStack(1085, 275, 1.0);
tinyVisualTokens(1255, 580, 3);
arrow(1545, 420, 1765, 420, C.black, 4, 26);
polyArrow([[1605, 420], [1605, 660]], C.black, 4, 24);
polyArrow([[1180, 530], [1180, 800], [1505, 800]], C.gold, 4, 22);
polyArrow([[1325, 635], [1325, 815]], C.gold, 4, 22);
arrow(1325, 455, 1325, 395, C.gold, 4, 22);
polyArrow([[1015, 445], [1065, 445]], C.black, 4, 23);

// AWRM module
addShape("roundRect", 900, 875, 460, 415, {
  fill: "#E9F8DF",
  line: { fill: C.green, width: 3 },
});
addShape("roundRect", 950, 900, 225, 95, {
  fill: C.greenFill,
  line: { fill: C.green, width: 3 },
});
addText("Text-Visual\nCorrelation", 970, 910, 185, 75, { fontSize: 31 });
arrow(1168, 948, 1266, 948, C.black, 4, 22);
tokenColumn(1262, 890, 4, C.amber, 0.9);
addShape("roundRect", 930, 1065, 270, 95, {
  fill: C.greenFill,
  line: { fill: C.green, width: 3 },
});
addText("Sentence-level\nSemantics", 958, 1075, 215, 75, { fontSize: 31 });
arrow(1200, 1112, 1265, 1112, C.black, 4, 22);
tokenColumn(1265, 1057, 4, C.tokenGreen, 0.9);
addLine(930, 948, 912, 948, C.black, 4);
addLine(912, 948, 912, 1185, C.black, 4);
addLine(912, 1185, 980, 1185, C.black, 4);
arrowHead(980, 1185, "right", 22, C.black);
addLine(1060, 1160, 1060, 1252, C.black, 4);
arrowHead(1060, 1160, "up", 22, C.black);
addShape("ellipse", 990, 1156, 55, 55, { fill: "#F3F9ED", line: { fill: C.black, width: 4 } });
addText("Fₜ", 995, 1161, 45, 45, { fontSize: 30, italic: true });
addText("Δₜ\nFₜ ⊕ (αΔₜ) = F′ₜ\nΔₜ", 1080, 1150, 245, 130, { fontSize: 32 });
addLine(1060, 1252, 1328, 1252, C.black, 4);
addLine(1328, 1252, 1328, 1112, C.black, 4);
label("AWRM: Attentive Word-level\nResidual Modulation", 900, 1402, 465, 88, 37);
arrow(855, 1168, 900, 1168, C.black, 4, 24);

// Enhanced text features, initial queries, query generation
tokenColumn(1410, 1025, 4, C.tokenBlue, 1.08);
addText("Enhanced\nText\nFeatures\nF′ₜ", 1350, 1198, 190, 150, { fontSize: 34 });
arrow(1360, 1112, 1410, 1112, C.black, 4, 22);
addLine(1450, 1112, 1515, 1112, C.black, 4);
polyArrow([[1515, 1112], [1515, 1015], [1580, 1015]], C.black, 4, 22);
addText("Initial Instance\nQueries Q₀", 1520, 1042, 230, 98, { fontSize: 33 });
addShape("ellipse", 1630, 1005, 38, 38, { fill: "#6EA3D3", line: { fill: "#1D5F8E", width: 3 } });
polyArrow([[1515, 1112], [1515, 1235], [1625, 1235]], C.black, 4, 22);
addShape("ellipse", 1620, 1222, 38, 38, { fill: "#B6D88B", line: { fill: C.green, width: 3 } });
arrow(1660, 1235, 1765, 1235, C.black, 4, 24);
addText("Reference\nPoints R₀", 1565, 1260, 240, 96, { fontSize: 33 });
addShape("roundRect", 1490, 650, 225, 300, {
  fill: C.greenFill2,
  line: { fill: C.green, width: 3 },
});
addText("Query\nGeneration\nModule", 1508, 728, 192, 145, { fontSize: 33, bold: true });
polyArrow([[1605, 650], [1605, 530]], C.black, 4, 23);
arrow(1715, 800, 1765, 800, C.black, 4, 24);
label("Query\nModule", 1568, 1405, 230, 82, 37);

// Multi-scale decoder
addShape("roundRect", 1765, 140, 235, 1248, {
  fill: "#FFDB62",
  line: { fill: C.gold, width: 3 },
});
addText("Cross-modal\nInteraction", 1778, 530, 205, 75, { fontSize: 35 });
addShape("circularArrow", 1820, 655, 130, 130, {
  fill: "#B9D39B",
  line: { fill: C.black, width: 3 },
});
addText("Multi-scale\nDecoder", 1795, 770, 190, 80, { fontSize: 35 });
addShape("circularArrow", 1820, 875, 130, 130, {
  fill: "#B9D39B",
  line: { fill: C.black, width: 3 },
  rotate: 180,
});
addText("Cross-modal\nInteraction", 1778, 980, 205, 75, { fontSize: 35 });
label("Multi-scale\nDecoder", 1760, 1405, 245, 82, 37);

// Prediction heads
addShape("roundRect", 2085, 140, 310, 590, {
  fill: "#FFF3C5",
  line: { fill: C.gold, width: 3 },
});
addText("Prediction\nHeads", 2135, 155, 205, 86, { fontSize: 38, bold: true });
const headBoxes = [
  ["Box\nPrediction", 2110, 260],
  ["Mask\nPrediction", 2110, 420],
  ["Query\nClassification", 2110, 575],
];
for (const [txt, x, y] of headBoxes) {
  addShape("roundRect", x, y, 255, 125, { fill: "#FFE39A", line: { fill: C.gold, width: 3 } });
  addText(txt, x + 20, y + 18, 215, 90, { fontSize: 36 });
}
arrow(2000, 800, 2085, 800, C.black, 4, 24);
polyArrow([[2000, 800], [2045, 800], [2045, 330], [2085, 330]], C.black, 4, 24);
polyArrow([[2045, 330], [2045, 480], [2085, 480]], C.black, 4, 24);
polyArrow([[2045, 480], [2045, 635], [2085, 635]], C.black, 4, 24);

// Existence branch
addShape("roundRect", 2085, 825, 310, 560, {
  fill: C.purpleFill,
  line: { fill: C.purple, width: 3 },
});
addText("Final Query\nFeatures H", 2050, 760, 330, 82, { fontSize: 35 });
for (let i = 0; i < 3; i += 1) tokenColumn(2165 + i * 45, 895, 2, i === 0 ? C.tokenBlue : "#A9B2E0", 0.9);
addShape("roundRect", 2105, 965, 250, 145, { fill: "#D6B6E8", line: { fill: C.purple, width: 3 } });
addText("Top-K\nForeground\nQueries", 2130, 985, 200, 105, { fontSize: 35 });
addShape("roundRect", 2105, 1165, 250, 112, { fill: "#D6B6E8", line: { fill: C.purple, width: 3 } });
addText("Weighted\nAggregation", 2125, 1180, 210, 82, { fontSize: 35 });
polyArrow([[2230, 1110], [2230, 1165]], C.black, 4, 22);
arrow(2355, 1220, 2410, 1220, C.black, 4, 24);
addShape("ellipse", 2400, 1198, 42, 42, { fill: "#D4A3D9", line: { fill: C.purple, width: 3 } });
addText("Existence\nProbability\npₑₓᵢₛₜ", 2355, 1242, 235, 115, { fontSize: 32 });
label("Existence\nBranch", 2110, 1404, 250, 82, 37);

// Right outputs
label("End-to-End Intuitive\nGrounding Output", 2410, 150, 395, 82, 38);
image("grounding_output.png", 2468, 268, 325, 300, "Grounding output crop");
addShape("rect", 2468, 268, 325, 300, { line: { fill: C.black, width: 3 } });
addShape("roundRect", 2490, 278, 275, 72, { fill: "#EFEFEF", line: { fill: C.black, width: 3 } });
addText("Turkey\n(standing on stool,)", 2486, 276, 290, 78, { fontSize: 31 });
addText("If pₑₓᵢₛₜ < τ", 2550, 620, 230, 60, { fontSize: 36, italic: true });
addShape("roundRect", 2538, 720, 250, 185, {
  fill: C.grayFill,
  line: { fill: C.black, width: 3 },
});
addText("Entire Final\nGrounding\nLogic", 2562, 738, 205, 140, { fontSize: 35 });
addText("τ", 2420, 812, 45, 45, { fontSize: 36, italic: true });
image("no_target.png", 2576, 1022, 210, 205, "No target match crop");
addShape("rect", 2576, 1022, 210, 205, { line: { fill: C.black, width: 3 } });
addText("If pₑₓᵢₛₜ < τ", 2550, 950, 230, 60, { fontSize: 36, italic: true });
addText("No target\nmatch found", 2510, 1228, 330, 92, { fontSize: 35 });
addText("(semantic mismatch)", 2552, 1324, 260, 45, { fontSize: 27 });
label("End-to-End Intuitive\nGrounding Output", 2405, 1405, 395, 82, 37);

// Output arrows and final logic connections
arrow(2395, 330, 2468, 330, C.black, 4, 26);
polyArrow([[2365, 480], [2430, 480], [2430, 720], [2538, 720]], C.black, 4, 24);
polyArrow([[2365, 635], [2430, 635], [2430, 760], [2538, 760]], C.black, 4, 24);
polyArrow([[2442, 1220], [2485, 1220], [2485, 840], [2538, 840]], C.black, 4, 24);
arrow(2465, 812, 2538, 812, C.black, 4, 24);
arrow(2663, 720, 2663, 568, C.black, 4, 24);
arrow(2663, 1022, 2663, 905, C.black, 4, 24);

// Cross-branch connector from decoder into existence branch
polyArrow([[2000, 800], [2040, 800], [2040, 1100], [2085, 1100]], C.black, 4, 24);

// Fine visual details for image stacks
addShape("rect", 74, 295, 145, 145, { fill: "#B9D2D8", line: { fill: C.black, width: 2 } });
image("primary_top.png", 86, 305, 150, 140, "Primary input overlay crop");
addBox(88, 312, 132, 118, "#F3C750", C.black, 3, false);
image("primary_top.png", 98, 323, 112, 98, "Primary input inner crop");

// Put the bottom input image stack back above the panel.
addShape("rect", 72, 910, 145, 160, { fill: "#B9D2D8", line: { fill: C.black, width: 2 } });
image("primary_single.png", 83, 920, 136, 147, "Primary sample overlay crop");
addBox(88, 925, 128, 130, "#F3C750", C.black, 3, false);
image("primary_single.png", 98, 937, 104, 106, "Primary sample inner crop");

function findInvalidNumbers(value, trail = "root", hits = []) {
  if (typeof value === "number" && !Number.isFinite(value)) {
    hits.push(trail);
    return hits;
  }
  if (Array.isArray(value)) {
    value.forEach((item, idx) => findInvalidNumbers(item, `${trail}[${idx}]`, hits));
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      findInvalidNumbers(item, `${trail}.${key}`, hits);
    }
  }
  return hits;
}

const invalidNumbers = findInvalidNumbers(presentation.toProto());
if (invalidNumbers.length) {
  throw new Error(`Invalid numeric values before export:\n${invalidNumbers.slice(0, 20).join("\n")}`);
}

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(pptxPath);

const pngBlob = await slide.export({ format: "png" });
await saveBlob(pngBlob, previewPath);

const patchResult = spawnSync(PYTHON, [path.join(root, "tools", "patch_pptx_media.py"), pptxPath, assetDir], {
  stdio: "inherit",
});
if (patchResult.status !== 0) {
  throw new Error(`patch_pptx_media.py failed with status ${patchResult.status}`);
}

const reloaded = await PresentationFile.importPptx(await fs.promises.readFile(pptxPath));
const reopenPngBlob = await reloaded.export({ format: "png" });
await saveBlob(reopenPngBlob, reopenPreviewPath);

console.log(JSON.stringify({
  pptxPath,
  previewPath,
  reopenPreviewPath,
  slideSize: { width: W, height: H },
}, null, 2));

process.exit(0);
