const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak
} = require("docx");

// ── helpers ──────────────────────────────────────────────────────────
const FONT = { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" };
const FONT_BOLD = { ...FONT, bold: true };

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function txt(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: 22, ...opts });
}
function txtB(text, opts = {}) {
  return txt(text, { font: FONT_BOLD, bold: true, ...opts });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: FONT_BOLD, size: 32, bold: true })],
  });
}
function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, font: FONT_BOLD, size: 28, bold: true })],
  });
}

function para(textOrRuns, opts = {}) {
  const children = typeof textOrRuns === "string"
    ? [txt(textOrRuns)]
    : textOrRuns;
  return new Paragraph({ spacing: { after: 120 }, ...opts, children });
}

function bulletItem(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [txt(text)],
  });
}

function numberedItem(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [txt(text)],
  });
}

// Table helpers
function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2B579A", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, font: FONT_BOLD, size: 20, bold: true, color: "FFFFFF" })] })],
  });
}

function dataCell(textOrRuns, width, opts = {}) {
  const children = typeof textOrRuns === "string"
    ? [txt(textOrRuns, { size: 20 })]
    : textOrRuns.map(t => typeof t === "string" ? txt(t, { size: 20 }) : t);
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ spacing: { after: 0 }, children })],
  });
}

function makeRow(cells, opts = {}) {
  return new TableRow({ cantSplit: true, ...opts, children: cells });
}

// ── Build document ───────────────────────────────────────────────────
const content = [];

// ── Cover / Title ────────────────────────────────────────────────────
content.push(
  new Paragraph({ spacing: { before: 2400 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "SpotZoom v4.0", font: FONT_BOLD, size: 52, bold: true, color: "2B579A" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [new TextRun({ text: "\u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", font: FONT_BOLD, size: 44, bold: true, color: "2B579A" })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [txt("\u65E5\u671F: 2026-05-12", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [txt("\u9879\u76EE: \u5149\u6591\u95ED\u73AF\u5BF9\u51C6\u7CFB\u7EDF (8821L)", { size: 24 })] }),
  new Paragraph({ children: [new PageBreak()] })
);

// ══════════════════════════════════════════════════════════════════════
// 一、本轮新增创新模块 (v4.0)
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u4E00\u3001\u672C\u8F6E\u65B0\u589E\u521B\u65B0\u6A21\u5757 (v4.0)"));

const mW = [450, 1400, 1400, 1800, 1800, 1800];

content.push(
  new Table({
    width: { size: 8650, type: WidthType.DXA },
    columnWidths: mW,
    rows: [
      makeRow([
        headerCell("#", mW[0]),
        headerCell("\u6A21\u5757\u540D", mW[1]),
        headerCell("\u7075\u611F\u6765\u6E90", mW[2]),
        headerCell("\u6838\u5FC3\u529F\u80FD", mW[3]),
        headerCell("\u5173\u952E\u7B97\u6CD5", mW[4]),
        headerCell("\u9884\u671F\u6536\u76CA", mW[5]),
      ]),
      makeRow([
        dataCell("1", mW[0], { shading: "F2F7FC" }),
        dataCell([txtB("WavefrontPredictor", { size: 18 }), txt(" (\u6CE2\u524D\u8BEF\u5DEE\u9884\u6D4B\u5668)", { size: 18 })], mW[1], { shading: "F2F7FC" }),
        dataCell("AOtools/HCIPy \u6CE2\u524D\u9884\u6D4B + LSTM \u9884\u6D4B\u7814\u7A76", mW[2], { shading: "F2F7FC" }),
        dataCell("\u57FA\u4E8E\u5386\u53F2\u8BEF\u5DEE\u5E8F\u5217\u9884\u6D4B\u672A\u6765\u5149\u6591\u504F\u79FB\uFF0C\u63D0\u4F9B\u524D\u9988\u8865\u507F", mW[3], { shading: "F2F7FC" }),
        dataCell("AR(p) \u81EA\u56DE\u5F52\u6A21\u578B + \u6B63\u5219\u5316\u6700\u5C0F\u4E8C\u4E58\u5728\u7EBF\u62DF\u5408", mW[4], { shading: "F2F7FC" }),
        dataCell("\u51CF\u5C11\u95ED\u73AF\u5EF6\u8FDF\u5F71\u54CD\uFF0C\u63D0\u5347\u6536\u655B\u901F\u5EA6 20-30%", mW[5], { shading: "F2F7FC" }),
      ]),
      makeRow([
        dataCell("2", mW[0]),
        dataCell([txtB("VibrationCompensator", { size: 18 }), txt(" (\u632F\u52A8\u68C0\u6D4B\u4E0E\u8865\u507F\u5668)", { size: 18 })], mW[1]),
        dataCell("BoT-SORT \u76F8\u673A\u8FD0\u52A8\u8865\u507F + ISO 11670", mW[2]),
        dataCell("\u5B9E\u65F6\u68C0\u6D4B\u5468\u671F\u6027/\u968F\u673A\u632F\u52A8\u5E76\u751F\u6210\u524D\u9988\u8865\u507F\u4FE1\u53F7", mW[3]),
        dataCell("FFT \u9891\u8C31\u5206\u6790 + IIR \u9677\u6CE2\u6EE4\u6CE2 + EMA \u57FA\u7EBF\u4F30\u8BA1", mW[4]),
        dataCell("\u6291\u5236\u73AF\u5883\u632F\u52A8\uFF0C\u63D0\u5347\u5BF9\u51C6\u7A33\u5B9A\u6027", mW[5]),
      ]),
      makeRow([
        dataCell("3", mW[0], { shading: "F2F7FC" }),
        dataCell([txtB("GaussianBeamFitter", { size: 18 }), txt(" (\u9AD8\u65AF\u5149\u675F\u62DF\u5408\u5668)", { size: 18 })], mW[1], { shading: "F2F7FC" }),
        dataCell("pyBeamProfiling + ISO 13694", mW[2], { shading: "F2F7FC" }),
        dataCell("2D \u9AD8\u65AF\u62DF\u5408\u7CBE\u786E\u63D0\u53D6\u5149\u675F\u53C2\u6570", mW[3], { shading: "F2F7FC" }),
        dataCell("Levenberg-Marquardt \u963B\u5C3C\u6700\u5C0F\u4E8C\u4E58 + \u56FE\u50CF\u77E9\u5FEB\u901F\u4F30\u8BA1", mW[4], { shading: "F2F7FC" }),
        dataCell("\u4E9A\u50CF\u7D20\u7EA7\u5149\u675F\u53C2\u6570\u63D0\u53D6\uFF0CM\u00B2 \u8D28\u91CF\u8BC4\u4F30", mW[5], { shading: "F2F7FC" }),
      ]),
      makeRow([
        dataCell("4", mW[0]),
        dataCell([txtB("EventBus", { size: 18 }), txt(" (\u4E8B\u4EF6\u603B\u7EBF\u7CFB\u7EDF)", { size: 18 })], mW[1]),
        dataCell("Bluesky RunEngine + Prometheus", mW[2]),
        dataCell("\u7EBF\u7A0B\u5B89\u5168\u7684\u53D1\u5E03-\u8BA2\u9605\u4E8B\u4EF6\u7CFB\u7EDF\uFF0C\u89E3\u8026\u6A21\u5757\u901A\u4FE1", mW[3]),
        dataCell("\u4F18\u5148\u7EA7\u961F\u5217 + \u5F02\u6B65\u5DE5\u4F5C\u7EBF\u7A0B + \u5F31\u5F15\u7528\u56DE\u8C03", mW[4]),
        dataCell("\u6A21\u5757\u89E3\u8026\uFF0C\u652F\u6301\u4E8B\u4EF6\u5F55\u5236\u56DE\u653E\u4E0E\u8C03\u8BD5", mW[5]),
      ]),
    ],
  })
);

// ══════════════════════════════════════════════════════════════════════
// 二、项目构建检测结果
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u4E8C\u3001\u9879\u76EE\u6784\u5EFA\u68C0\u6D4B\u7ED3\u679C"));

content.push(heading2("2.1 \u5BFC\u5165\u4E0E\u4F9D\u8D56"));
content.push(bulletItem("\u5EF6\u8FDF\u5BFC\u5165\u673A\u5236: \u6B63\u5E38 (\u6240\u6709 ML \u6A21\u5757\u901A\u8FC7 _import_ml_module \u5BB9\u9519\u5BFC\u5165)"));
content.push(bulletItem("\u5FAA\u73AF\u5BFC\u5165\u98CE\u9669: \u65E0 (22 \u4E2A ML \u5B50\u6A21\u5757\u5B8C\u5168\u72EC\u7ACB)"));
content.push(bulletItem("\u5916\u90E8\u4F9D\u8D56: numpy, opencv-python (\u5FC5\u9700); ultralytics, pyautogui, pywin32 (\u53EF\u9009)"));
content.push(bulletItem("\u65B0\u589E\u6A21\u5757\u96C6\u6210: __init__.py \u548C SpotZoom.py \u5DF2\u66F4\u65B0"));

content.push(heading2("2.2 \u4EE3\u7801\u7EDF\u8BA1"));
content.push(bulletItem("\u4E3B\u6587\u4EF6 SpotZoom.py: ~2600 \u884C"));
content.push(bulletItem("ML \u6A21\u5757\u5305: 22 \u4E2A .py \u6587\u4EF6 (\u542B 4 \u4E2A\u65B0\u589E)"));
content.push(bulletItem("\u603B\u5BFC\u51FA\u7B26\u53F7: 58 \u4E2A (__all__)"));
content.push(bulletItem("\u6587\u6863\u5B57\u7B26\u4E32\u8986\u76D6\u7387: ~95%"));

// ══════════════════════════════════════════════════════════════════════
// 三、问题检测汇总
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u4E09\u3001\u95EE\u9898\u68C0\u6D4B\u6C47\u603B"));

content.push(heading2("3.1 \u6309\u4E25\u91CD\u5EA6\u5206\u5E03"));

const sevW = [2000, 1500, 5500];
content.push(
  new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: sevW,
    rows: [
      makeRow([
        headerCell("\u4E25\u91CD\u5EA6", sevW[0]),
        headerCell("\u6570\u91CF", sevW[1]),
        headerCell("\u8BF4\u660E", sevW[2]),
      ]),
      makeRow([
        dataCell([txtB("P0 (\u7D27\u6025)", { size: 20, color: "CC0000" })], sevW[0], { shading: "FDECEC" }),
        dataCell([txtB("3", { size: 20, color: "CC0000" })], sevW[1], { shading: "FDECEC" }),
        dataCell("\u53EF\u80FD\u5BFC\u81F4\u7A0B\u5E8F\u5D29\u6E83\u6216\u786C\u4EF6\u72B6\u6001\u4E0D\u4E00\u81F4", sevW[2], { shading: "FDECEC" }),
      ]),
      makeRow([
        dataCell([txtB("P1 (\u91CD\u8981)", { size: 20, color: "E67E00" })], sevW[0], { shading: "FFF5E6" }),
        dataCell([txtB("4", { size: 20, color: "E67E00" })], sevW[1], { shading: "FFF5E6" }),
        dataCell("\u529F\u80FD\u7F3A\u5931\u6216\u6570\u636E\u4E22\u5931\u98CE\u9669", sevW[2], { shading: "FFF5E6" }),
      ]),
      makeRow([
        dataCell([txtB("P2 (\u4E2D\u7B49)", { size: 20, color: "1A7F37" })], sevW[0], { shading: "ECF5EC" }),
        dataCell([txtB("7", { size: 20, color: "1A7F37" })], sevW[1], { shading: "ECF5EC" }),
        dataCell("\u6027\u80FD\u95EE\u9898\u6216\u4EE3\u7801\u8D28\u91CF\u95EE\u9898", sevW[2], { shading: "ECF5EC" }),
      ]),
      makeRow([
        dataCell([txtB("P3 (\u4F4E)", { size: 20, color: "57606A" })], sevW[0], { shading: "F0F0F0" }),
        dataCell([txtB("5", { size: 20, color: "57606A" })], sevW[1], { shading: "F0F0F0" }),
        dataCell("\u4EE3\u7801\u89C4\u8303\u6216\u6587\u6863\u95EE\u9898", sevW[2], { shading: "F0F0F0" }),
      ]),
      makeRow([
        dataCell([txtB("\u5408\u8BA1", { size: 20 })], sevW[0]),
        dataCell([txtB("19", { size: 20 })], sevW[1]),
        dataCell("", sevW[2]),
      ]),
    ],
  })
);

// ── 3.2 P0 ──────────────────────────────────────────────────────────
content.push(heading2("3.2 P0 \u7D27\u6025\u95EE\u9898\u8BE6\u60C5"));

function issueBlock(num, title, file, line, desc, impact, fix) {
  return [
    para([txtB("\u95EE\u9898 " + num + ": " + title, { size: 22, color: "CC0000" })], { spacing: { before: 200, after: 80 } }),
    bulletItem("\u6587\u4EF6: " + file + ", \u884C\u53F7: " + line),
    bulletItem("\u63CF\u8FF0: " + desc),
    bulletItem("\u5F71\u54CD: " + impact),
    bulletItem("\u4FEE\u590D: " + fix),
  ];
}

content.push(...issueBlock(
  1, "Z \u8F74\u79FB\u52A8\u540E\u5F02\u5E38\u4E0D\u56DE\u6EDA",
  "SpotZoom.py", "1727-1740",
  "run() \u65B9\u6CD5\u4E2D Z \u8F74\u4E0A\u79FB\u540E\u5982\u679C _detect_center_with_retry \u629B\u51FA\u5F02\u5E38\uFF0CZ \u8F74\u4E0D\u4F1A\u56DE\u79FB\uFF0C\u5BFC\u81F4\u786C\u4EF6\u72B6\u6001\u4E0D\u4E00\u81F4",
  "\u786C\u4EF6\u72B6\u6001\u4E0D\u4E00\u81F4\uFF0C\u4E0B\u6B21\u8FD0\u884C\u65F6 Z \u8F74\u4F4D\u7F6E\u504F\u79FB",
  "\u5728 Z \u8F74\u79FB\u52A8\u5468\u56F4\u6DFB\u52A0 try/finally \u4FDD\u8BC1\u56DE\u6EDA"
));

content.push(...issueBlock(
  2, "parse_args() \u8FD4\u56DE int \u5BFC\u81F4 AttributeError",
  "SpotZoom.py", "2449-2461",
  "\u914D\u7F6E\u6587\u4EF6\u52A0\u8F7D\u5931\u8D25\u65F6 parse_args() \u8FD4\u56DE int(1) \u800C\u975E Namespace\uFF0C\u540E\u7EED args.log_level \u5BFC\u81F4 AttributeError",
  "\u914D\u7F6E\u6587\u4EF6\u8DEF\u5F84\u9519\u8BEF\u65F6\u7A0B\u5E8F\u5D29\u6E83\u5E76\u7ED9\u51FA\u4EE4\u4EBA\u56F0\u60D1\u7684\u9519\u8BEF\u4FE1\u606F",
  "\u4F7F\u7528 sys.exit(1) \u66FF\u4EE3 return 1"
));

content.push(...issueBlock(
  3, "\u4FE1\u53F7\u5904\u7406\u65E0\u6548\uFF0CCtrl+C \u65E0\u6CD5\u4E2D\u65AD",
  "SpotZoom.py", "2472-2484",
  "_shutdown_requested \u662F main() \u5C40\u90E8\u53D8\u91CF\uFF0C\u672A\u4F20\u9012\u7ED9 SpotZoomController\uFF0C\u6240\u6709\u5BF9\u9F50\u5FAA\u73AF\u4E2D\u90FD\u6CA1\u6709\u68C0\u67E5 shutdown \u6807\u5FD7",
  "\u7528\u6237\u6309 Ctrl+C \u540E\u7A0B\u5E8F\u7EE7\u7EED\u8FD0\u884C\uFF0CKeyboardInterrupt \u6C38\u8FDC\u4E0D\u4F1A\u88AB\u629B\u51FA",
  "\u5C06 shutdown \u6807\u5FD7\u4F20\u9012\u7ED9 controller\uFF0C\u5728\u5FAA\u73AF\u4E2D\u68C0\u67E5"
));

// ── 3.3 P1 ──────────────────────────────────────────────────────────
content.push(heading2("3.3 P1 \u91CD\u8981\u95EE\u9898\u8BE6\u60C5"));

content.push(...issueBlock(
  4, "\u6240\u6709\u521B\u65B0\u6A21\u5757\u65E0\u6CD5\u901A\u8FC7\u547D\u4EE4\u884C\u542F\u7528",
  "SpotZoom.py", "2333-2548",
  "AlignmentConfig \u5B9A\u4E49\u4E86 40+ \u521B\u65B0\u6A21\u5757\u914D\u7F6E\u5B57\u6BB5\uFF0C\u4F46 parse_args() \u4E2D\u6CA1\u6709\u5BF9\u5E94\u7684 add_argument\uFF0Cmain() \u4E2D\u4E5F\u6CA1\u6709\u4F20\u9012",
  "\u6240\u6709\u521B\u65B0\u6A21\u5757\u5728\u547D\u4EE4\u884C\u4F7F\u7528\u4E2D\u6C38\u8FDC\u65E0\u6CD5\u88AB\u542F\u7528",
  "\u5728 parse_args() \u4E2D\u6DFB\u52A0 --enable-kalman, --enable-subpixel \u7B49\u53C2\u6570"
));

content.push(...issueBlock(
  5, "\u5927\u91CF\u6A21\u5757\u8D44\u6E90\u672A\u5728 close() \u4E2D\u91CA\u653E",
  "SpotZoom.py", "2086-2114",
  "TrajectoryRecorder, SafetyManager, ActiveLearningCollector \u7B49 12 \u4E2A\u6A21\u5757\u672A\u5728 close() \u4E2D\u6E05\u7406",
  "\u8F68\u8FF9\u6570\u636E\u672A\u5BFC\u51FA\uFF0C\u5B89\u5168\u72B6\u6001\u672A\u7EC8\u7ED3\uFF0C\u91C7\u96C6\u6570\u636E\u53EF\u80FD\u4E22\u5931",
  "\u5728 close() \u4E2D\u9010\u4E00\u8C03\u7528\u5404\u6A21\u5757\u7684 reset()/close()/end_run()"
));

content.push(...issueBlock(
  6, "Kalman \u8FD4\u56DE None \u5BFC\u81F4 TypeError",
  "SpotZoom.py", "1889-1898",
  "_smooth_detection \u4E2D Kalman \u8FD4\u56DE\u503C\u672A\u505A None \u68C0\u67E5\uFF0C\u4E0E\u4E9A\u50CF\u7D20\u90E8\u5206\u7684\u6B63\u786E\u5904\u7406\u4E0D\u4E00\u81F4",
  "\u5982\u679C Kalman \u5185\u90E8\u77E9\u9635\u5947\u5F02\u8FD4\u56DE None\uFF0C\u540E\u7EED\u8BBF\u95EE center[0] \u4F1A TypeError",
  "\u6DFB\u52A0 if smoothed is not None \u68C0\u67E5"
));

content.push(...issueBlock(
  7, "active_learning_collector.py \u53D8\u91CF\u8BEF\u7528",
  "active_learning_collector.py", "247",
  "sample_reason=reason or should\uFF0Cshould \u662F bool \u7C7B\u578B\uFF0C\u5F53 reason \u4E3A\u7A7A\u65F6 sample_reason \u53D8\u6210 \"True\"/\"False\"",
  "\u751F\u6210\u7684\u6570\u636E\u6807\u6CE8\u4E2D sample_reason \u5B57\u6BB5\u65E0\u610F\u4E49",
  "\u6539\u4E3A reason or \"routine_collect\""
));

// ── 3.4 P2 ──────────────────────────────────────────────────────────
content.push(heading2("3.4 P2 \u4E2D\u7B49\u95EE\u9898\u8BE6\u60C5"));

function issueBrief(num, title, file, line, desc, fix) {
  return [
    para([txtB("\u95EE\u9898 " + num + ": " + title, { size: 22 })], { spacing: { before: 160, after: 60 } }),
    bulletItem("\u6587\u4EF6: " + file + (line ? ", \u884C\u53F7: " + line : "")),
    bulletItem("\u63CF\u8FF0: " + desc),
    bulletItem("\u4FEE\u590D: " + fix),
  ];
}

content.push(...issueBrief(
  8, "anomaly_detector.py \u548C realtime_performance_monitor.py \u4F7F\u7528 List+\u624B\u52A8\u622A\u65AD",
  "anomaly_detector.py, realtime_performance_monitor.py", "",
  "\u591A\u5904\u4F7F\u7528 List + if len > N: list = list[-N:] \u6A21\u5F0F\uFF0C\u5E94\u4F7F\u7528 deque(maxlen=N)\uFF0C\u6BCF\u6B21\u622A\u65AD\u521B\u5EFA\u65B0\u5217\u8868\uFF0CO(N) \u65F6\u95F4\u590D\u6742\u5EA6",
  "\u66FF\u6362\u4E3A collections.deque(maxlen=N)"
));

content.push(...issueBrief(
  9, "vibration_compensator.py \u7C7B\u53D8\u91CF\u5BFC\u81F4\u5B9E\u4F8B\u95F4\u72B6\u6001\u6C61\u67D3",
  "vibration_compensator.py", "603-604",
  "_notch_coeffs_cache \u548C _notch_freq_cache \u662F\u7C7B\u53D8\u91CF\uFF0C\u591A\u5B9E\u4F8B\u5171\u4EAB\uFF0C\u591A\u5B9E\u4F8B\u573A\u666F\u4E0B\u6EE4\u6CE2\u5668\u7CFB\u6570\u9519\u8BEF",
  "\u6539\u4E3A\u5B9E\u4F8B\u53D8\u91CF self._notch_coeffs_cache"
));

content.push(...issueBrief(
  10, "active_learning_collector.py YOLO \u6807\u7B7E\u786C\u7F16\u7801\u56FE\u50CF\u5C3A\u5BF8",
  "active_learning_collector.py", "414",
  "img_w, img_h = 640, 480 \u786C\u7F16\u7801\uFF0C\u5B9E\u9645\u56FE\u50CF\u5C3A\u5BF8\u53EF\u80FD\u4E0D\u540C\uFF0CYOLO \u6807\u7B7E\u5F52\u4E00\u5316\u9519\u8BEF",
  "\u4ECE\u5B9E\u9645 frame.shape \u83B7\u53D6\u5C3A\u5BF8"
));

content.push(...issueBrief(
  11, "PID \u63A7\u5236\u5668\u6CE8\u91CA\u4E0E\u5B9E\u73B0\u4E0D\u4E00\u81F4",
  "SpotZoom.py", "228-232",
  "\u6CE8\u91CA\u58F0\u79F0 \"derivative on measurement\" \u4F46\u5B9E\u9645\u5B9E\u73B0\u662F \"derivative on error\"\uFF0C\u4EE3\u7801\u503A\u52A1\uFF0C\u5C06\u6765\u4FEE\u6539 setpoint \u65F6\u4F1A\u8E29\u5751",
  "\u4FEE\u6B63\u6CE8\u91CA\u6216\u4FEE\u6539\u5B9E\u73B0"
));

content.push(...issueBrief(
  12, "Recovery scan \u4F7F\u7528\u8FC7\u65F6\u7684 _last_frame",
  "SpotZoom.py", "1985",
  "_smooth_detection \u5728 recovery scan \u4E2D\u4F7F\u7528\u8FC7\u65F6\u7684 self._last_frame\uFF0C\u4E9A\u50CF\u7D20\u5B9A\u4F4D\u4F7F\u7528\u9519\u8BEF\u56FE\u50CF\u6570\u636E",
  "\u5728 recovery scan \u4E2D\u66F4\u65B0 self._last_frame"
));

content.push(...issueBrief(
  13, "ThorlabsXYStage/XPSZAxis \u6784\u9020\u51FD\u6570\u8D44\u6E90\u6CC4\u6F0F",
  "SpotZoom.py", "1159-1165, 1475-1489",
  "\u6784\u9020\u51FD\u6570\u4E2D\u8BBE\u5907 open \u540E\u5982\u679C\u540E\u7EED\u6B65\u9AA4\u5931\u8D25\uFF0C\u5DF2\u6253\u5F00\u7684\u8BBE\u5907\u4E0D\u4F1A\u88AB\u5173\u95ED\uFF0C\u8BBE\u5907\u53E5\u67C4\u6CC4\u6F0F",
  "\u4F7F\u7528 try/except \u5305\u88F9\u5E76\u5728\u5931\u8D25\u65F6\u8C03\u7528 close()"
));

content.push(...issueBrief(
  14, "SpotZoom.py __all__ \u9057\u6F0F ML \u6A21\u5757\u7C7B",
  "SpotZoom.py", "116-149",
  "ZernikeAberrationAnalyzer, ImageJacobianController \u7B49 5 \u4E2A\u7C7B\u672A\u5728 __all__ \u4E2D\u58F0\u660E\uFF0CAPI \u6587\u6863\u4E0D\u5B8C\u6574",
  "\u8865\u5168 __all__ \u5217\u8868"
));

// ── 3.5 P3 ──────────────────────────────────────────────────────────
content.push(heading2("3.5 P3 \u4F4E\u4F18\u5148\u7EA7\u95EE\u9898"));

content.push(bulletItem("\u95EE\u9898 15: event_bus.py \u5BFC\u5165 heapq \u672A\u4F7F\u7528"));
content.push(bulletItem("\u95EE\u9898 16: \u591A\u4E2A\u6A21\u5757\u5BFC\u5165 Deque \u7C7B\u578B\u4F46\u672A\u4F7F\u7528 (beam_stability_analyzer, focus_search, realtime_performance_monitor)"));
content.push(bulletItem("\u95EE\u9898 17: \u8BEF\u5DEE\u5E45\u5EA6\u8BA1\u7B97\u516C\u5F0F\u5728 3 \u4E2A\u6A21\u5757\u4E2D\u91CD\u590D"));
content.push(bulletItem("\u95EE\u9898 18: rl_environment.py _trajectory \u548C _reward_history \u65E0\u6700\u5927\u957F\u5EA6\u9650\u5236"));
content.push(bulletItem("\u95EE\u9898 19: classic_spot_detector.py \u4E2D SpotDetection \u4E0E SpotZoom.py \u4E2D\u91CD\u590D\u5B9A\u4E49"));

// ══════════════════════════════════════════════════════════════════════
// 四、代码质量评估
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u56DB\u3001\u4EE3\u7801\u8D28\u91CF\u8BC4\u4F30"));

content.push(heading2("4.1 \u4F18\u70B9"));
content.push(bulletItem("\u6A21\u5757\u5316\u8BBE\u8BA1\u4F18\u79C0: 22 \u4E2A ML \u5B50\u6A21\u5757\u5B8C\u5168\u89E3\u8026\uFF0C\u96F6\u4EA4\u53C9\u5F15\u7528"));
content.push(bulletItem("\u5EF6\u8FDF\u5BFC\u5165\u5BB9\u9519: ML \u6A21\u5757\u4E0D\u53EF\u7528\u65F6\u4E0D\u5F71\u54CD\u6838\u5FC3\u529F\u80FD"));
content.push(bulletItem("\u8D44\u6E90\u7BA1\u7406: \u786C\u4EF6\u9A71\u52A8\u4F7F\u7528\u4E0A\u4E0B\u6587\u7BA1\u7406\u5668\uFF0C\u5D4C\u5957 try/finally \u4FDD\u8BC1\u91CA\u653E"));
content.push(bulletItem("\u6587\u6863\u8986\u76D6\u7387\u9AD8: ~95% \u7684\u516C\u5171 API \u6709\u5B8C\u6574\u6587\u6863\u5B57\u7B26\u4E32"));
content.push(bulletItem("\u63A5\u53E3\u4E00\u81F4\u6027: Protocol \u62BD\u8C61\u786C\u4EF6\u63A5\u53E3\uFF0C\u652F\u6301\u591A\u79CD\u540E\u7AEF\u70ED\u5207\u6362"));

content.push(heading2("4.2 \u5F85\u6539\u8FDB"));
content.push(bulletItem("\u5F02\u5E38\u5904\u7406: 54 \u5904 except Exception \u8FC7\u4E8E\u5BBD\u6CDB"));
content.push(bulletItem("\u6570\u636E\u7ED3\u6784: \u591A\u5904\u5E94\u4F7F\u7528 deque \u66FF\u4EE3 list"));
content.push(bulletItem("\u7C7B\u578B\u6807\u6CE8: \u90E8\u5206\u6A21\u5757\u8FD4\u56DE dict \u5E94\u6539\u4E3A TypedDict"));
content.push(bulletItem("\u547D\u4EE4\u884C\u96C6\u6210: \u521B\u65B0\u6A21\u5757\u914D\u7F6E\u5B57\u6BB5\u65E0\u6CD5\u901A\u8FC7 CLI \u8BBE\u7F6E"));

// ══════════════════════════════════════════════════════════════════════
// 五、性能评估
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u4E94\u3001\u6027\u80FD\u8BC4\u4F30"));

content.push(bulletItem("\u4E3B\u5FAA\u73AF\u5EF6\u8FDF: \u68C0\u6D4B(~15ms YOLO) + settle(350ms) + \u7535\u673A\u79FB\u52A8(~50ms) \u2248 415ms/\u5468\u671F"));
content.push(bulletItem("\u5185\u5B58\u4F7F\u7528: ML \u6A21\u5757\u603B\u8BA1\u7EA6 5-10MB (\u5168\u90E8\u542F\u7528\u65F6)"));
content.push(bulletItem("\u74F6\u9888: YOLO \u63A8\u7406\u5EF6\u8FDF\u662F\u4E3B\u8981\u74F6\u9888\uFF0C\u5EFA\u8BAE\u4F7F\u7528 TensorRT/ONNX \u52A0\u901F"));

// ══════════════════════════════════════════════════════════════════════
// 六、下一步优化方案
// ══════════════════════════════════════════════════════════════════════
content.push(heading1("\u516D\u3001\u4E0B\u4E00\u6B65\u4F18\u5316\u65B9\u6848"));

content.push(heading2("\u77ED\u671F (1-2\u5929)"));
content.push(numberedItem("\u4FEE\u590D P0 \u7D27\u6025\u95EE\u9898 (Z \u8F74\u56DE\u6EDA\u3001parse_args\u3001\u4FE1\u53F7\u5904\u7406)", "short"));
content.push(numberedItem("\u4FEE\u590D P1 \u91CD\u8981\u95EE\u9898 (CLI \u53C2\u6570\u3001close() \u6E05\u7406\u3001Kalman None \u68C0\u67E5)", "short"));
content.push(numberedItem("\u8865\u5168 __all__ \u5217\u8868", "short"));

content.push(heading2("\u4E2D\u671F (1\u5468)"));
content.push(numberedItem("\u5C06 List+\u624B\u52A8\u622A\u65AD\u66FF\u6362\u4E3A deque", "mid"));
content.push(numberedItem("\u4FEE\u590D\u7C7B\u53D8\u91CF\u72B6\u6001\u6C61\u67D3", "mid"));
content.push(numberedItem("\u4FEE\u590D YOLO \u6807\u7B7E\u786C\u7F16\u7801\u5C3A\u5BF8", "mid"));
content.push(numberedItem("\u6DFB\u52A0\u5355\u5143\u6D4B\u8BD5\u6846\u67B6 (pytest)", "mid"));

content.push(heading2("\u957F\u671F (1\u6708)"));
content.push(numberedItem("\u5B9E\u73B0\u521B\u65B0\u6A21\u5757\u7684 CLI \u53C2\u6570\u96C6\u6210", "long"));
content.push(numberedItem("\u6DFB\u52A0 TensorRT/ONNX \u63A8\u7406\u540E\u7AEF", "long"));
content.push(numberedItem("\u5B9E\u73B0 EventBus \u8DE8\u6A21\u5757\u4E8B\u4EF6\u8054\u52A8", "long"));
content.push(numberedItem("\u6DFB\u52A0 Web \u4EEA\u8868\u76D8 (\u53EF\u9009)", "long"));

// ── Assemble ────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 22 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT_BOLD, color: "2B579A" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT_BOLD, color: "2B579A" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1, keepNext: false, keepLines: false },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "short",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "mid",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "long",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [txt("SpotZoom v4.0 \u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", { size: 18, color: "888888" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            txt("\u7B2C ", { size: 18, color: "888888" }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "888888" }),
            txt(" \u9875", { size: 18, color: "888888" }),
          ],
        })],
      }),
    },
    children: content,
  }],
});

// ── Write ───────────────────────────────────────────────────────────
const OUTPUT = "e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v4_\u7EFC\u5408\u68C0\u6D4B\u62A5\u544A_20260512.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("OK: " + OUTPUT);
}).catch(err => {
  console.error("FAIL:", err);
  process.exit(1);
});
