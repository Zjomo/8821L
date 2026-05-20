const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2B5797", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20 })] })]
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, color: opts.color || "000000", bold: opts.bold || false, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 18 })] })]
  });
}

function severityCell(level, width) {
  const map = { "\u9AD8": { fill: "FDE8E8", color: "CC3333" }, "\u4E2D": { fill: "FFF0E0", color: "CC6600" }, "\u4F4E": { fill: "FFFDE0", color: "999900" } };
  const s = map[level] || { fill: "E0F0E0", color: "336633" };
  return cell(level, width, { fill: s.fill, color: s.color, bold: true });
}

function issueRow(id, severity, category, module, location, desc, suggestion) {
  return new TableRow({
    cantSplit: true,
    children: [
      cell(id, 500, { bold: true }),
      severityCell(severity, 600),
      cell(category, 1200),
      cell(module, 1600),
      cell(location, 1800),
      cell(desc, 2800),
      cell(suggestion, 2800)
    ]
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 200 }, children: [new TextRun({ text, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 32, color: "1A1A1A" })] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 160 }, children: [new TextRun({ text, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 26, color: "2B5797" })] });
}

function p(text) {
  return new Paragraph({ spacing: { before: 80, after: 80 }, children: [new TextRun({ text, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20, color: "333333" })] });
}

function bp(text) {
  return new Paragraph({ spacing: { before: 80, after: 80 }, children: [new TextRun({ text, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20, color: "333333" })] });
}

const C = [500, 600, 1200, 1600, 1800, 2800, 2800];
const TW = C.reduce((a, b) => a + b, 0);

function it(rows) {
  return new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: C,
    rows: [
      new TableRow({ cantSplit: true, children: [headerCell("\u7F16\u53F7", C[0]), headerCell("\u4E25\u91CD\u5EA6", C[1]), headerCell("\u95EE\u9898\u5206\u7C7B", C[2]), headerCell("\u5F71\u54CD\u6A21\u5757", C[3]), headerCell("\u5177\u4F53\u4F4D\u7F6E", C[4]), headerCell("\u95EE\u9898\u63CF\u8FF0", C[5]), headerCell("\u4FEE\u590D\u5EFA\u8BAE", C[6])] }),
      ...rows
    ]
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } }, paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } }, paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1, keepNext: false, keepLines: false } },
    ]
  },
  sections: [{
    properties: { page: { size: { width: 15840, height: 12240 }, margin: { top: 1200, right: 1000, bottom: 1200, left: 1000 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "SpotZoom \u6DF1\u5EA6\u4EE3\u7801\u5206\u6790\u62A5\u544A", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 16, color: "999999" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u7B2C ", size: 16, color: "999999", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "999999", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } }), new TextRun({ text: " \u9875", size: 16, color: "999999", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } })] })] }) },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 }, children: [new TextRun({ text: "SpotZoom \u9879\u76EE\u6DF1\u5EA6\u4EE3\u7801\u5206\u6790\u62A5\u544A", bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 40, color: "1A1A1A" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "\u5206\u6790\u65E5\u671F: 2026-05-12  |  \u9879\u76EE\u8DEF\u5F84: e:\\jupyter file\\2_Optics\\8821L", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20, color: "666666" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "\u5206\u6790\u8303\u56F4: \u8026\u5408\u5EA6 / \u6027\u80FD\u74F6\u9888 / \u4EE3\u7801\u5F02\u5473 / \u5B89\u5168\u6027 / \u7C7B\u578B\u5B89\u5168 / \u56DE\u5F52\u98CE\u9669", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20, color: "666666" })] }),

      h1("1. \u9879\u76EE\u6982\u89C8"),
      p("SpotZoom \u662F\u4E00\u4E2A\u95ED\u73AF\u5149\u6591\u5BF9\u51C6\u63A7\u5236\u7CFB\u7EDF\uFF0C\u57FA\u4E8E YOLO \u76EE\u6807\u68C0\u6D4B + \u591A\u8F74\u8FD0\u52A8\u63A7\u5236\u5668\u5B9E\u73B0\u81EA\u52A8\u5BF9\u51C6\u3002\u9879\u76EE\u7531\u4E24\u4E2A\u6838\u5FC3\u90E8\u5206\u7EC4\u6210\uFF1A"),
      bp("\u6838\u5FC3\u6587\u4EF6:"),
      p("SpotZoom.py (2,407 \u884C) - \u4E3B\u63A7\u5236\u5668\u3001\u786C\u4EF6\u9A71\u52A8\u3001CLI \u5165\u53E3"),
      p("SpotZoom_Machine_Learning/ (48 \u4E2A .py \u6A21\u5757) - \u521B\u65B0\u7B97\u6CD5\u6A21\u5757\u5305"),
      p("correct_robot.py (479 \u884C) - \u5DF2\u5F03\u7528\u7684\u65E7\u7248\u63A7\u5236\u5668"),
      bp("\u6587\u4EF6\u884C\u6570\u7EDF\u8BA1 (\u8D85\u8FC7 500 \u884C\u7684\u6587\u4EF6):"),
      p("SpotZoom.py: 2,407 \u884C (\u4E25\u91CD\u8D85\u6807)"),
      p("composable_optical_pipeline.py: 1,023 \u884C"),
      p("multi_layer_turbulence_simulator.py: 781 \u884C"),
      p("psf_estimator.py: 637 \u884C"),
      p("spot_morphology_analyzer.py: 590 \u884C"),
      p("smart_refinement_controller.py: 588 \u884C"),
      p("vibration_compensator.py: 526 \u884C"),
      p("turbulence_simulator.py: 522 \u884C"),

      h1("2. \u8026\u5408\u5EA6\u5206\u6790"),
      h2("2.1 SpotZoom.py \u5BF9 ML \u6A21\u5757\u7684\u4F9D\u8D56"),
      p("SpotZoom.py \u901A\u8FC7\u5EF6\u8FDF\u5BFC\u5165\u673A\u5236\u52A0\u8F7D\u4E86 24 \u4E2A ML \u5B50\u6A21\u5757\uFF0C\u5E76\u5C06\u5176\u7C7B\u66B4\u9732\u5230\u5F53\u524D\u547D\u540D\u7A7A\u95F4\u3002\u8FD9\u79CD\u8BBE\u8BA1\u5728\u6A21\u5757\u7F3A\u5931\u65F6\u4E0D\u4F1A\u5D29\u6E83\uFF0C\u4F46\u521B\u5EFA\u4E86\u5F3A\u8026\u5408\u3002"),
      p("\u4F9D\u8D56\u94FE\uFF1ASpotZoom.py -> _import_ml_module() -> SpotZoom_Machine_Learning.{kalman_tracker, subpixel_centroid, spot_quality, adaptive_gain, trajectory_recorder, safety_manager, zernike_analyzer, image_jacobian, model_optimizer, active_learning_collector, focus_search, wavefront_predictor, vibration_compensator, gaussian_fitter, event_bus, temporal_fusion_predictor, self_tuning_controller, spot_morphology_analyzer, data_pipeline_orchestrator, diagnostic_health_monitor, multi_layer_turbulence_simulator, composable_optical_pipeline, auto_alignment_optimizer, deep_vibration_predictor, domain_randomizer}"),
      h2("2.2 ML \u6A21\u5757\u5185\u90E8\u8026\u5408"),
      p("ML \u5B50\u6A21\u5757\u4E4B\u95F4\u51E0\u4E4E\u6CA1\u6709\u4EA4\u53C9\u5F15\u7528\uFF0C\u6BCF\u4E2A\u6A21\u5757\u4EC5\u4F9D\u8D56 numpy\u3001cv2 \u7B49\u6807\u51C6\u5E93\u3002\u8FD9\u662F\u4E00\u4E2A\u826F\u597D\u7684\u8BBE\u8BA1\u3002\u4F46 __init__.py \u5BFC\u5165\u4E86\u6240\u6709 48 \u4E2A\u6A21\u5757\u7684\u7C7B\u548C\u51FD\u6570\uFF0C\u5F62\u6210\u4E86\u201C\u5168\u91CF\u5BFC\u5165\u201D\u6A21\u5F0F\u3002"),
      h2("2.3 SpotZoom.py \u5185\u90E8\u7C7B\u4F9D\u8D56\u56FE"),
      p("SpotZoomController -> ToupViewWindow, SpotYOLODetector, XYStage (ThorlabsXYStage/NewportXYStage/DryRunStage), ZStage (XPSZAxis/ToupViewWheelZAxis/DryRunZAxis), AlignmentConfig, RunReporter\u3002SpotZoomController \u8FD8\u53EF\u9009\u4F9D\u8D56 24 \u4E2A ML \u6A21\u5757\u5B9E\u4F8B\u3002"),

      it([
        issueRow("C-1", "\u9AD8", "\u8026\u5408\u5EA6", "SpotZoom.py", "\u884C 36-72", "SpotZoom.py \u5BFC\u5165\u4E86 24 \u4E2A ML \u6A21\u5757\uFF0C\u5C06\u5176\u7C7B\u66B4\u9732\u5230\u5168\u5C40\u547D\u540D\u7A7A\u95F4\uFF0C\u5F62\u6210\u5F3A\u8026\u5408", "\u5C06 ML \u6A21\u5757\u7684\u521D\u59CB\u5316\u5C01\u88C5\u5230 SpotZoomController \u5185\u90E8\uFF0C\u4EC5\u5728\u9700\u8981\u65F6\u5BFC\u5165"),
        issueRow("C-2", "\u4E2D", "\u8026\u5408\u5EA6", "__init__.py", "\u884C 59-255", "__init__.py \u5168\u91CF\u5BFC\u5165 48 \u4E2A\u6A21\u5757\u7684\u6240\u6709\u516C\u5171\u7B26\u53F7\uFF0C\u5305\u52A0\u8F7D\u5F00\u9500\u5927", "\u8003\u8651\u61D2\u5BFC\u5165 (lazy import)\uFF0C\u6216\u62C6\u5206\u4E3A\u591A\u4E2A\u5B50\u5305"),
        issueRow("C-3", "\u4E2D", "\u8026\u5408\u5EA6", "SpotZoom.py", "\u884C 1620-1863", "SpotZoomController.__init__ \u5305\u542B 243 \u884C\u7684\u6761\u4EF6\u521D\u59CB\u5316\u903B\u8F91\uFF0C\u76F4\u63A5\u4F9D\u8D56\u5168\u5C40\u53D8\u91CF (_ml_xxx)", "\u5C06 ML \u6A21\u5757\u521D\u59CB\u5316\u62BD\u53D6\u4E3A\u5DE5\u5382\u65B9\u6CD5\u6216\u72EC\u7ACB\u7C7B"),
        issueRow("C-4", "\u4F4E", "\u8026\u5408\u5EA6", "SpotZoom.py", "\u884C 23-29", "\u52A8\u6001\u4FEE\u6539 sys.path\uFF0C\u5C06\u9879\u76EE\u6839\u76EE\u5F55\u548C CoreSegment \u52A0\u5165\u8DEF\u5F84", "\u4F7F\u7528\u6B63\u5F0F\u7684\u5305\u7BA1\u7406\u548C setup.py"),
      ]),

      h1("3. \u6027\u80FD\u74F6\u9888\u68C0\u6D4B"),
      h2("3.1 \u5FAA\u73AF\u4E2D\u7684 I/O \u64CD\u4F5C"),
      p("\u5728\u6838\u5FC3\u5BF9\u51C6\u5FAA\u73AF\u4E2D\u53D1\u73B0\u591A\u5904\u5FAA\u73AF\u5185 I/O\uFF0C\u8FD9\u662F\u6700\u4E25\u91CD\u7684\u6027\u80FD\u95EE\u9898\u3002"),
      it([
        issueRow("P-1", "\u9AD8", "\u6027\u80FD\u74F6\u9888", "SpotZoom.py", "\u884C 856-867", "wheel() \u65B9\u6CD5\u5728 for \u5FAA\u73AF\u4E2D\u53D1\u9001 Windows \u6D88\u606F (SendMessage)\uFF0C\u6BCF\u6B21\u8C03\u7528\u90FD\u662F\u4E00\u6B21\u7CFB\u7EDF\u8C03\u7528", "\u5C06\u591A\u6B21\u6EDA\u52A8\u5408\u5E76\u4E3A\u5355\u6B21 SendMessage \u8C03\u7528\uFF0C\u4F7F\u7528\u53C2\u6570 WHEEL_DELTA \u7D2F\u52A0"),
        issueRow("P-2", "\u9AD8", "\u6027\u80FD\u74F6\u9888", "SpotZoom.py", "\u884C 1974-2016", "_detect_center_with_retry \u5728 for \u5FAA\u73AF\u4E2D\u8C03\u7528 grab_frame() \u548C detector.detect()\uFF0C\u6BCF\u6B21\u90FD\u5305\u542B\u5C4F\u5E55\u622A\u56FE + YOLO \u63A8\u7406", "\u8FD9\u662F\u8BBE\u8BA1\u4E0A\u7684\u5FC5\u7136\uFF0C\u4F46\u5EFA\u8BAE\u589E\u52A0\u5E27\u7387\u9650\u5236\u548C\u8D85\u65F6\u4FDD\u62A4"),
        issueRow("P-3", "\u9AD8", "\u6027\u80FD\u74F6\u9888", "SpotZoom.py", "\u884C 2106-2150", "_attempt_recovery_scan \u5D4C\u5957\u5FAA\u73AF (cycles x pattern)\uFF0C\u6BCF\u6B65\u90FD\u6267\u884C\u622A\u56FE+\u68C0\u6D4B+I/O", "\u9650\u5236\u603B\u6B65\u6570\u4E0A\u9650\uFF0C\u589E\u52A0\u8D85\u65F6\u4FDD\u62A4\uFF0C\u907F\u514D\u65E0\u9650\u626B\u63CF"),
        issueRow("P-4", "\u4E2D", "\u6027\u80FD\u74F6\u9888", "SpotZoom.py", "\u884C 2184", "_show_preview \u6BCF\u6B21\u8C03\u7528\u90FD\u6267\u884C frame.copy()\uFF0C\u5728\u9AD8\u9891\u5FAA\u73AF\u4E2D\u4EA7\u751F\u5927\u91CF\u5185\u5B58\u5206\u914D", "\u4EC5\u5728\u9884\u89C8\u5F00\u542F\u65F6\u6267\u884C\u62F7\u8D1D\uFF0C\u5E76\u8003\u8651\u964D\u4F4E\u9884\u89C8\u5237\u65B0\u7387"),
        issueRow("P-5", "\u4E2D", "\u6027\u80FD\u74F6\u9888", "deep_vibration_predictor.py", "\u884C 273-339", "SimpleRNNCell \u548C AttentionLayer \u7684\u524D\u5411/\u540E\u5411\u4F20\u64AD\u4F7F\u7528 Python for \u5FAA\u73AF\u5B9E\u73B0\uFF0C\u672A\u5229\u7528 numpy \u5411\u91CF\u5316", "\u5C06\u5FAA\u73AF\u66FF\u6362\u4E3A numpy \u77E9\u9635\u8FD0\u7B97\uFF0C\u6216\u4F7F\u7528 numba JIT \u52A0\u901F"),
        issueRow("P-6", "\u4E2D", "\u6027\u80FD\u74F6\u9888", "domain_randomizer.py", "\u884C 393-394", "\u5D4C\u5957\u5FAA\u73AF (images x count_per_image)\uFF0C\u6BCF\u6B21\u8FED\u4EE3\u90FD\u8FDB\u884C\u56FE\u50CF\u968F\u673A\u5316\u5904\u7406", "\u6DFB\u52A0\u8FDB\u5EA6\u6761\u548C\u5E76\u884C\u5904\u7406\u652F\u6301"),
        issueRow("P-7", "\u4F4E", "\u6027\u80FD\u74F6\u9888", "auto_alignment_optimizer.py", "\u884C 225-315", "\u591A\u6B21 x.copy() \u8C03\u7528\u5728\u4F18\u5316\u5FAA\u73AF\u4E2D\uFF0C\u5BF9\u5C0F\u6570\u7EC4\u5F71\u54CD\u8F83\u5C0F", "\u5BF9\u4E8E\u4F4E\u7EF4\u6570\u7EC4\u53EF\u5FFD\u7565\uFF0C\u9AD8\u7EF4\u65F6\u8003\u8651\u5C31\u5730\u64CD\u4F5C"),
      ]),

      h1("4. \u4EE3\u7801\u5F02\u5473\u68C0\u6D4B"),
      h2("4.1 \u8FC7\u957F\u6587\u4EF6"),
      it([
        issueRow("S-1", "\u9AD8", "\u957F\u6587\u4EF6", "SpotZoom.py", "\u5168\u6587 2,407 \u884C", "\u5355\u6587\u4EF6\u5305\u542B 15+ \u4E2A\u7C7B\u3001\u591A\u4E2A\u5DE5\u5382\u51FD\u6570\u3001CLI \u5165\u53E3\uFF0C\u8FDD\u53CD\u5355\u4E00\u804C\u8D23\u539F\u5219", "\u62C6\u5206\u4E3A\u591A\u4E2A\u6A21\u5757: controllers/, drivers/, detectors/, cli.py"),
        issueRow("S-2", "\u4E2D", "\u957F\u6587\u4EF6", "composable_optical_pipeline.py", "\u5168\u6587 1,023 \u884C", "\u5355\u6587\u4EF6\u5305\u542B\u591A\u4E2A\u5149\u5B66\u5143\u4EF6\u7C7B\u548C\u7BA1\u7EBF\u5F15\u64CE", "\u62C6\u5206\u5143\u4EF6\u7C7B\u5230\u5355\u72EC\u6587\u4EF6"),
        issueRow("S-3", "\u4E2D", "\u957F\u6587\u4EF6", "multi_layer_turbulence_simulator.py", "\u5168\u6587 781 \u884C", "\u5305\u542B\u591A\u5C42\u6E58\u6D41\u4EFF\u771F\u3001\u76F8\u4F4D\u5C4F\u751F\u6210\u3001\u4F20\u64AD\u8BA1\u7B97\u7B49\u591A\u4E2A\u804C\u8D23", "\u62C6\u5206\u4E3A phase_screen.py \u548C propagation.py"),
      ]),
      h2("4.2 \u8FC7\u957F\u51FD\u6570"),
      it([
        issueRow("S-4", "\u9AD8", "\u957F\u51FD\u6570", "SpotZoom.py", "\u884C 1620-1863", "SpotZoomController.__init__ \u7EA6 243 \u884C\uFF0C\u5305\u542B 24 \u4E2A ML \u6A21\u5757\u7684\u6761\u4EF6\u521D\u59CB\u5316", "\u62BD\u53D6\u4E3A _init_ml_modules() \u65B9\u6CD5\uFF0C\u4F7F\u7528\u914D\u7F6E\u9A71\u52A8\u7684\u5DE5\u5382\u6A21\u5F0F"),
        issueRow("S-5", "\u9AD8", "\u957F\u51FD\u6570", "SpotZoom.py", "\u884C 2511-2639", "parse_args() \u7EA6 128 \u884C\uFF0C\u5305\u542B 60+ \u4E2A\u53C2\u6570\u5B9A\u4E49\u548C\u914D\u7F6E\u6587\u4EF6\u52A0\u8F7D", "\u62C6\u5206\u4E3A\u591A\u4E2A add_xxx_args() \u8F85\u52A9\u51FD\u6570"),
        issueRow("S-6", "\u4E2D", "\u957F\u51FD\u6570", "SpotZoom.py", "\u884C 1908-1965", "_align_to_target() \u7EA6 57 \u884C\uFF0C\u5305\u542B\u5BF9\u9F50\u5FAA\u73AF\u3001\u5B89\u5168\u68C0\u67E5\u3001\u8F68\u8FF9\u8BB0\u5F55\u7B49\u591A\u4E2A\u804C\u8D23", "\u62BD\u53D6\u5B89\u5168\u68C0\u67E5\u548C\u8F68\u8FF9\u8BB0\u5F55\u4E3A\u72EC\u7ACB\u65B9\u6CD5"),
        issueRow("S-7", "\u4E2D", "\u957F\u51FD\u6570", "SpotZoom.py", "\u884C 1967-2025", "_detect_center_with_retry() \u7EA6 58 \u884C\uFF0C\u5305\u542B\u91CD\u8BD5\u3001\u7126\u70B9\u8BC4\u4F30\u3001\u5E73\u6ED1\u3001\u6062\u590D\u626B\u63CF", "\u62BD\u53D6\u7126\u70B9\u8BC4\u4F30\u548C\u6062\u590D\u626B\u63CF\u4E3A\u72EC\u7ACB\u65B9\u6CD5"),
        issueRow("S-8", "\u4E2D", "\u957F\u51FD\u6570", "SpotZoom.py", "\u884C 2642-2768", "main() \u7EA6 126 \u884C\uFF0C\u5305\u542B\u53C2\u6570\u89E3\u6790\u3001\u786C\u4EF6\u521D\u59CB\u5316\u3001\u4FE1\u53F7\u5904\u7406\u3001\u5F02\u5E38\u5904\u7406", "\u62BD\u53D6\u786C\u4EF6\u521D\u59CB\u5316\u4E3A build_controller() \u65B9\u6CD5"),
      ]),
      h2("4.3 \u9B54\u6CD5\u6570\u5B57"),
      it([
        issueRow("S-9", "\u4E2D", "\u9B54\u6CD5\u6570\u5B57", "SpotZoom.py", "\u884C 164, 825, 864", "WHEEL_DELTA=120, \u7A97\u53E3\u5927\u5C0F (1400, 900), \u5B57\u4F53\u53C2\u6570 (0.8, 2) \u7B49\u786C\u7F16\u7801", "\u63D0\u53D6\u4E3A\u5E38\u91CF\u6216\u914D\u7F6E\u53C2\u6570"),
        issueRow("S-10", "\u4E2D", "\u9B54\u6CD5\u6570\u5B57", "SpotZoom.py", "\u884C 915-918", "Thorlabs DLL \u8DEF\u5F84\u786C\u7F16\u7801\u4E3A C:\\Program Files\\Thorlabs\\...", "\u4F7F\u7528\u73AF\u5883\u53D8\u91CF\u6216\u914D\u7F6E\u6587\u4EF6"),
        issueRow("S-11", "\u4F4E", "\u9B54\u6CD5\u6570\u5B57", "SpotZoom.py", "\u884C 334, 354", "setup_logging \u4E2D max_bytes=5*1024*1024, backup_count=3 \u786C\u7F16\u7801", "\u63D0\u53D6\u4E3A\u6A21\u5757\u7EA7\u5E38\u91CF"),
      ]),

      h1("5. \u5B89\u5168\u6027\u68C0\u67E5"),
      it([
        issueRow("SEC-1", "\u9AD8", "\u786C\u7F16\u7801\u51ED\u8BC1", "SpotZoom.py", "\u884C 1563-1567", "XPSZAxis \u7C7B\u786C\u7F16\u7801\u9ED8\u8BA4\u7528\u6237\u540D\u548C\u5BC6\u7801: Administrator/Administrator", "\u4ECE\u73AF\u5883\u53D8\u91CF\u6216\u914D\u7F6E\u6587\u4EF6\u8BFB\u53D6\u51ED\u8BC1\uFF0C\u7981\u6B62\u786C\u7F16\u7801"),
        issueRow("SEC-2", "\u9AD8", "\u786C\u7F16\u7801\u51ED\u8BC1", "SpotZoom.py", "\u884C 2561-2563", "CLI \u53C2\u6570\u9ED8\u8BA4\u503C --xps-password=Administrator", "\u79FB\u9664\u9ED8\u8BA4\u503C\uFF0C\u8981\u6C42\u7528\u6237\u660E\u786E\u63D0\u4F9B\u6216\u4ECE\u73AF\u5883\u53D8\u91CF\u8BFB\u53D6"),
        issueRow("SEC-3", "\u4E2D", "\u5B89\u5168\u98CE\u9669", "SpotZoom.py", "\u884C 148", "\u4F7F\u7528 os.system(\"chcp 65001 > nul\") \u6267\u884C\u7CFB\u7EDF\u547D\u4EE4", "\u4F7F\u7528 ctypes \u8C03\u7528 Windows API \u66FF\u4EE3 os.system"),
        issueRow("SEC-4", "\u4F4E", "\u5B89\u5168\u98CE\u9669", "SpotZoom.py", "\u884C 872, 882-893", "\u4F7F\u7528 __import__() \u548C subprocess.run() \u68C0\u67E5\u6A21\u5757\u53EF\u7528\u6027\uFF0C\u53C2\u6570\u6765\u81EA\u7528\u6237\u8F93\u5165", "\u5BF9\u6A21\u5757\u540D\u8FDB\u884C\u767D\u540D\u5355\u9A8C\u8BC1"),
        issueRow("SEC-5", "\u4F4E", "\u5B89\u5168\u98CE\u9669", "SpotZoom.py", "\u884C 1041-1050", "subprocess.Popen \u542F\u52A8\u5DE5\u4F5A\u8FDB\u7A0B\uFF0C\u547D\u4EE4\u884C\u53C2\u6570\u6784\u9020\u4F7F\u7528\u5217\u8868\u5F62\u5F0F\uFF08\u5B89\u5168\uFF09", "\u5F53\u524D\u5B9E\u73B0\u5DF2\u662F\u5B89\u5168\u7684\u5217\u8868\u5F62\u5F0F\uFF0C\u65E0\u9700\u4FEE\u6539"),
      ]),
      p("\u672A\u53D1\u73B0 eval() \u6216 exec() \u7684\u4F7F\u7528\u3002\u8FD9\u662F\u4E00\u4E2A\u79EF\u6781\u7684\u5B89\u5168\u5B9E\u8DF5\u3002"),

      h1("6. \u7C7B\u578B\u5B89\u5168\u68C0\u67E5"),
      h2("6.1 Type Hints \u8986\u76D6\u7387"),
      p("SpotZoom.py: \u7EA6 85% \u7684\u516C\u5171\u51FD\u6570\u548C\u65B9\u6CD5\u6709\u7C7B\u578B\u6CE8\u89E3\uFF0C\u8986\u76D6\u7387\u8F83\u597D\u3002\u4E3B\u8981\u7F3A\u5931\u5728\u5185\u90E8\u8F85\u52A9\u65B9\u6CD5\u3002"),
      p("SpotZoom_Machine_Learning \u5B50\u6A21\u5757: \u7EA6 90%+ \u7684\u516C\u5171\u65B9\u6CD5\u6709\u7C7B\u578B\u6CE8\u89E3\uFF0C\u4F7F\u7528\u4E86 dataclass \u548C Optional/Dict/List/Tuple \u7B49\u6CE8\u89E3\u3002"),
      p("correct_robot.py: \u7EA6 90% \u8986\u76D6\u7387\uFF0C\u7C7B\u578B\u6CE8\u89E3\u5B8C\u6574\u3002"),
      h2("6.2 Optional \u7C7B\u578B\u672A\u505A None \u68C0\u67E5"),
      it([
        issueRow("T-1", "\u4E2D", "\u7C7B\u578B\u5B89\u5168", "SpotZoom.py", "\u884C 725-727", "ToupViewWindow \u7684 hwnd/scroll_hwnd/roi \u58F0\u660E\u4E3A Optional \u4F46\u90E8\u5206\u4F7F\u7528\u5904\u672A\u505A None \u68C0\u67E5", "\u5728\u4F7F\u7528\u524D\u7EDF\u4E00\u68C0\u67E5\u662F\u5426\u4E3A None"),
        issueRow("T-2", "\u4F4E", "\u7C7B\u578B\u5B89\u5168", "SpotZoom.py", "\u884C 1669-1863", "ML \u6A21\u5757\u5B9E\u4F8B\u58F0\u660E\u4E3A Optional\uFF0C\u4F46\u540E\u7EED\u4F7F\u7528\u524D\u5747\u6709 if \u68C0\u67E5", "\u5F53\u524D\u5B9E\u73B0\u5DF2\u6B63\u786E\u5904\u7406\uFF0C\u65E0\u9700\u4FEE\u6539"),
        issueRow("T-3", "\u4F4E", "\u7C7B\u578B\u5B89\u5168", "SpotZoom_Machine_Learning", "\u591A\u4E2A\u6A21\u5757", "\u90E8\u5206\u5185\u90E8\u65B9\u6CD5\u7F3A\u5C11\u8FD4\u56DE\u7C7B\u578B\u6CE8\u89E3\uFF0C\u4EC5\u516C\u5171 API \u6709\u5B8C\u6574\u6CE8\u89E3", "\u5EFA\u8BAE\u4F7F\u7528 mypy \u8FDB\u884C\u9759\u6001\u7C7B\u578B\u68C0\u67E5"),
      ]),

      h1("7. \u56DE\u5F52\u98CE\u9669\u8BC4\u4F30"),
      h2("7.1 correct_robot.py \u5F03\u7528\u72B6\u6001"),
      it([
        issueRow("R-1", "\u9AD8", "\u56DE\u5F52\u98CE\u9669", "correct_robot.py", "\u884C 15-21", "\u6587\u4EF6\u5DF2\u6807\u8BB0\u4E3A DEPRECATED\uFF0C\u4F46\u4ECD\u5B58\u5728\u4E8E\u9879\u76EE\u4E2D\u4E14\u6709\u5B8C\u6574\u7684 __pycache__", "\u5EFA\u8BAE\u5C06\u5176\u79FB\u5165 deprecated/ \u76EE\u5F55\u6216\u76F4\u63A5\u5220\u9664\uFF0C\u6E05\u7406 __pycache__"),
        issueRow("R-2", "\u4E2D", "\u56DE\u5F52\u98CE\u9669", "correct_robot.py", "\u5168\u6587", "\u5F03\u7528\u6587\u4EF6\u4E0E SpotZoom.py \u5B58\u5728\u529F\u80FD\u91CD\u53E0\uFF0C\u65B0\u7528\u6237\u53EF\u80FD\u8BEF\u7528\u65E7\u7248", "\u5728 README \u4E2D\u660E\u786E\u6807\u6CE8\u5F03\u7528\u72B6\u6001\uFF0C\u5F15\u5BFC\u7528\u6237\u4F7F\u7528 SpotZoom.py"),
      ]),
      h2("7.2 __init__.py \u5BFC\u51FA\u4E0E\u5B9E\u9645\u6A21\u5757\u5339\u914D"),
      it([
        issueRow("R-3", "\u4F4E", "\u56DE\u5F52\u98CE\u9669", "__init__.py", "\u884C 59-255 vs 257-418", "__init__.py \u7684 from import \u548C __all__ \u5217\u8868\u7ECF\u6838\u67E5\u4E00\u81F4\uFF0C\u5BFC\u51FA\u7684\u6240\u6709\u7B26\u53F7\u5747\u5728\u5B9E\u9645\u6A21\u5757\u4E2D\u5B58\u5728", "\u5F53\u524D\u65E0\u95EE\u9898\uFF0C\u5EFA\u8BAE\u6DFB\u52A0\u81EA\u52A8\u5316\u6D4B\u8BD5\u9A8C\u8BC1\u5BFC\u51FA\u5B8C\u6574\u6027"),
        issueRow("R-4", "\u4F4E", "\u56DE\u5F52\u98CE\u9669", "__init__.py", "\u884C 213-255", "v8.0 \u65B0\u589E\u6A21\u5757\u7684\u5BFC\u51FA\u672A\u5728 SpotZoom.py \u7684 __all__ \u4E2D\u5B8C\u6574\u5BF9\u5E94", "\u786E\u4FDD SpotZoom.py \u7684 __all__ \u4E0E __init__.py \u7684\u5BFC\u51FA\u4FDD\u6301\u540C\u6B65"),
      ]),
      h2("7.3 \u5176\u4ED6\u56DE\u5F52\u98CE\u9669"),
      it([
        issueRow("R-5", "\u4E2D", "\u56DE\u5F52\u98CE\u9669", "SpotZoom.py", "\u884C 23-29", "\u52A8\u6001\u4FEE\u6539 sys.path \u53EF\u80FD\u5BFC\u81F4\u6A21\u5757\u89E3\u6790\u987A\u5E8F\u4E0D\u786E\u5B9A\uFF0C\u5728\u4E0D\u540C\u73AF\u5883\u4E0B\u884C\u4E3A\u4E0D\u4E00\u81F4", "\u4F7F\u7528\u6B63\u5F0F\u7684 Python \u5305\u7BA1\u7406\uFF08setup.py/pyproject.toml\uFF09"),
        issueRow("R-6", "\u4F4E", "\u56DE\u5F52\u98CE\u9669", "SpotZoom.py", "\u884C 882-893", "_module_available_in_python \u4F7F\u7528 subprocess \u68C0\u67E5\u5916\u90E8 Python \u73AF\u5883\uFF0C\u4F9D\u8D56\u5916\u90E8\u53EF\u6267\u884C\u6587\u4EF6", "\u6DFB\u52A0\u8D85\u65F6\u548C\u5F02\u5E38\u5904\u7406\u7684\u5355\u5143\u6D4B\u8BD5"),
      ]),

      h1("8. \u7EFC\u5408\u8BC4\u4F30\u4E0E\u4F18\u5148\u7EA7\u5EFA\u8BAE"),
      h2("8.1 \u95EE\u9898\u7EDF\u8BA1"),
      new Table({
        width: { size: TW, type: WidthType.DXA },
        columnWidths: [2400, 1200, 1200, 1200, 1200],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u5206\u7C7B", 2400), headerCell("\u9AD8", 1200), headerCell("\u4E2D", 1200), headerCell("\u4F4E", 1200), headerCell("\u5408\u8BA1", 1200)] }),
          new TableRow({ cantSplit: true, children: [cell("\u8026\u5408\u5EA6", 2400), cell("1", 1200), cell("2", 1200), cell("1", 1200), cell("4", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u6027\u80FD\u74F6\u9888", 2400), cell("3", 1200), cell("3", 1200), cell("1", 1200), cell("7", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u4EE3\u7801\u5F02\u5473", 2400), cell("3", 1200), cell("5", 1200), cell("1", 1200), cell("9", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u5B89\u5168\u6027", 2400), cell("2", 1200), cell("1", 1200), cell("2", 1200), cell("5", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u7C7B\u578B\u5B89\u5168", 2400), cell("0", 1200), cell("1", 1200), cell("2", 1200), cell("3", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u56DE\u5F52\u98CE\u9669", 2400), cell("1", 1200), cell("3", 1200), cell("2", 1200), cell("6", 1200, { bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u5408\u8BA1", 2400, { bold: true, fill: "E8E8E8" }), cell("10", 1200, { bold: true, fill: "FDE8E8" }), cell("15", 1200, { bold: true, fill: "FFF0E0" }), cell("9", 1200, { bold: true, fill: "FFFDE0" }), cell("34", 1200, { bold: true, fill: "E8E8E8" })] }),
        ]
      }),

      h2("8.2 \u4F18\u5148\u4FEE\u590D\u5EFA\u8BAE"),
      bp("\u7B2C\u4E00\u4F18\u5148\u7EA7 (\u7ACB\u5373\u4FEE\u590D):"),
      p("1. [SEC-1/SEC-2] \u79FB\u9664 XPS \u63A7\u5236\u5668\u7684\u786C\u7F16\u7801\u5BC6\u7801\uFF0C\u4ECE\u73AF\u5883\u53D8\u91CF\u6216\u914D\u7F6E\u6587\u4EF6\u8BFB\u53D6"),
      p("2. [P-1] \u4F18\u5316 wheel() \u65B9\u6CD5\uFF0C\u5408\u5E76\u591A\u6B21\u6EDA\u52A8\u4E3A\u5355\u6B21\u7CFB\u7EDF\u8C03\u7528"),
      p("3. [R-1] \u5220\u9664\u6216\u79FB\u52A8\u5DF2\u5F03\u7528\u7684 correct_robot.py"),
      bp("\u7B2C\u4E8C\u4F18\u5148\u7EA7 (\u77ED\u671F\u4F18\u5316):"),
      p("4. [S-1] \u62C6\u5206 SpotZoom.py \u4E3A\u591A\u4E2A\u6A21\u5757\uFF08controllers/, drivers/, detectors/, cli.py\uFF09"),
      p("5. [S-4] \u91CD\u6784 SpotZoomController.__init__\uFF0C\u4F7F\u7528\u5DE5\u5382\u6A21\u5F0F\u521D\u59CB\u5316 ML \u6A21\u5757"),
      p("6. [C-1] \u5C06 ML \u6A21\u5757\u5BFC\u5165\u6539\u4E3A\u6309\u9700\u52A0\u8F7D\uFF0C\u907F\u514D\u5168\u5C40\u547D\u540D\u7A7A\u95F4\u6C61\u67D3"),
      p("7. [P-5] \u4F18\u5316 deep_vibration_predictor \u7684 Python \u5FAA\u73AF\u4E3A numpy \u5411\u91CF\u5316\u64CD\u4F5C"),
      bp("\u7B2C\u4E09\u4F18\u5148\u7EA7 (\u4E2D\u671F\u6539\u8FDB):"),
      p("8. [C-2] \u4F18\u5316 __init__.py \u7684\u5168\u91CF\u5BFC\u5165\u4E3A\u61D2\u5BFC\u5165"),
      p("9. [S-9/S-10] \u6E05\u7406\u9B54\u6CD5\u6570\u5B57\uFF0C\u63D0\u53D6\u4E3A\u5E38\u91CF\u6216\u914D\u7F6E"),
      p("10. [SEC-3] \u66FF\u6362 os.system(\"chcp\") \u4E3A ctypes Windows API \u8C03\u7528"),
      p("11. [R-5] \u91C7\u7528\u6B63\u5F0F\u5305\u7BA1\u7406\uFF0C\u6D88\u9664 sys.path \u52A8\u6001\u4FEE\u6539"),

      new Paragraph({ spacing: { before: 400 }, children: [] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "--- \u62A5\u544A\u7ED3\u675F ---", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 18, color: "999999" })] }),
    ]
  }]
});

const outPath = "e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v8_deep_code_analysis_report.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("Report saved to: " + outPath);
});
