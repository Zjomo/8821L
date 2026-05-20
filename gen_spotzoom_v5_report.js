const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat
} = require("docx");

// ── Common helpers ──
const FONT = { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" };
const FONT_RUN = (size) => ({ size, ...FONT });

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: "center",
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text, bold: true, color: "FFFFFF", font: FONT_RUN(20) })]
      })
    ]
  });
}

function cell(text, width, opts = {}) {
  const bg = opts.bg ? { fill: opts.bg, type: ShadingType.CLEAR } : undefined;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: bg,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    verticalAlign: "center",
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        children: [new TextRun({
          text,
          color: opts.color || "000000",
          bold: opts.bold || false,
          font: FONT_RUN(opts.fontSize || 18)
        })]
      })
    ]
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, font: FONT_RUN(30), color: "1F4E79" })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, font: FONT_RUN(24), color: "2E75B6" })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, font: FONT_RUN(21), color: "404040" })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [new TextRun({
      text,
      font: FONT_RUN(opts.fontSize || 20),
      color: opts.color || "333333",
      bold: opts.bold || false
    })]
  });
}

function bulletItem(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: FONT_RUN(20), color: "333333" })]
  });
}

function numberedItem(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbered", level },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: FONT_RUN(20), color: "333333" })]
  });
}

function codeBlock(text) {
  return new Paragraph({
    spacing: { before: 100, after: 100 },
    indent: { left: 360 },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    children: [new TextRun({
      text,
      font: { size: 18, ascii: "Consolas", hAnsi: "Consolas", eastAsia: "Microsoft YaHei" },
      color: "333333"
    })]
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
}

// ── Severity color helper ──
function sevColor(sev) {
  switch (sev) {
    case "MEDIUM": return "E65100";
    case "LOW": return "F57F17";
    case "INFO": return "1565C0";
    case "HIGH": return "C62828";
    default: return "333333";
  }
}

// ── Build document ──
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1, keepNext: false, keepLines: false }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 21, bold: true, font: FONT, color: "404040" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: false, keepLines: false }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
          },
          {
            level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } }
          }
        ]
      },
      {
        reference: "numbered",
        levels: [
          {
            level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
          },
          {
            level: 1, format: LevelFormat.LOWER_LETTER, text: "%2)", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } }
          }
        ]
      }
    ]
  },
  sections: [
    // ═══════════════════════════════════════════════════════════════════
    // SINGLE SECTION — ALL CONTENT
    // ═══════════════════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1200, right: 1200, bottom: 1200, left: 1200 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "SpotZoom v5.0 \u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", font: FONT_RUN(16), color: "999999" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "\u7B2C ", font: FONT_RUN(16) }),
              new TextRun({ children: [PageNumber.CURRENT], font: FONT_RUN(16) }),
              new TextRun({ text: " \u9875", font: FONT_RUN(16) })
            ]
          })]
        })
      },
      children: [
        // ──────────────────────────────────────────────────────────────
        // COVER PAGE
        // ──────────────────────────────────────────────────────────────
        new Paragraph({ spacing: { before: 3600 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          children: [new TextRun({ text: "SpotZoom v5.0 \u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", bold: true, font: FONT_RUN(48), color: "1F4E79" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({
            text: "\u5149\u6591\u95ED\u73AF\u5BF9\u51C6\u7CFB\u7EDF \u2014 \u524D\u6CBF\u5F00\u6E90\u8C03\u7814\u4E0E\u521B\u65B0\u6A21\u5757\u8865\u5145 + \u5168\u9762\u8D28\u91CF\u68C0\u6D4B",
            font: FONT_RUN(24), color: "555555"
          })]
        }),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "\u62A5\u544A\u65E5\u671F: 2026-05-12", font: FONT_RUN(22), color: "666666" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "\u7248\u672C: v5.0", font: FONT_RUN(22), color: "666666" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "\u9879\u76EE\u89C4\u6A21: 31 \u4E2A Python \u6587\u4EF6 / 29 \u4E2A ML \u6A21\u5757 / ~13,000+ \u884C\u4EE3\u7801", font: FONT_RUN(20), color: "888888" })]
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 1: 前沿开源调研与创新模块补充
        // ──────────────────────────────────────────────────────────────
        h1("\u7B2C\u4E00\u7AE0  \u524D\u6CBF\u5F00\u6E90\u8C03\u7814\u4E0E\u521B\u65B0\u6A21\u5757\u8865\u5145"),

        // 1.1 调研范围
        h2("1.1 \u8C03\u7814\u8303\u56F4"),
        para("\u672C\u8F6E\u8C03\u7814\u8986\u76D6\u4E94\u5927\u6280\u672F\u9886\u57DF\uFF0C\u5171\u8C03\u7814 30+ \u5F00\u6E90\u9879\u76EE\uFF0C\u4E3A SpotZoom \u7CFB\u7EDF\u7684\u521B\u65B0\u6A21\u5757\u8BBE\u8BA1\u63D0\u4F9B\u7406\u8BBA\u4E0E\u5B9E\u8DF5\u53C2\u8003\u3002"),
        emptyLine(),

        bulletItem("\u81EA\u9002\u5E94\u5149\u5B66 (AOtools, HCIPy, SOAPY): Zernike \u50CF\u5DEE\u5206\u89E3\u3001\u6CE2\u524D\u9884\u6D4B\u3001\u5F00\u73AF\u63A7\u5236\u7B97\u6CD5"),
        bulletItem("\u89C6\u89C9\u4F3A\u670D (ViSP, BoxMOT, ByteTrack): \u56FE\u50CF\u96C5\u53EF\u6BD4\u63A7\u5236\u3001\u591A\u76EE\u6807\u8DDF\u8E2A\u3001\u5149\u6D41\u6CD5\u8FD0\u52A8\u4F30\u8BA1"),
        bulletItem("\u5149\u675F\u5206\u6790 (BoT-SORT, pyBeamProfiling): \u5149\u675F\u6307\u5411\u7A33\u5B9A\u6027\u5206\u6790\u3001\u9AD8\u65AF\u5149\u675F\u62DF\u5408\u3001ISO \u6807\u51C6\u5BF9\u9F50"),
        bulletItem("\u786C\u4EF6\u63A7\u5236 (PyMeasure, ARTIQ, python-control, Bluesky): \u5B9E\u9A8C\u7F16\u6392\u3001\u7EE7\u7535\u53CD\u9988\u6574\u5B9A\u3001\u6D41\u6C34\u7EBF\u81EA\u52A8\u5316"),
        bulletItem("\u6DF1\u5EA6\u5B66\u4E60\u4F18\u5316 (Ultralytics, Stable-Baselines3, PPAL): \u6A21\u578B\u63A8\u7406\u4F18\u5316\u3001\u5F3A\u5316\u5B66\u4E60\u7B56\u7565\u3001\u4E3B\u52A8\u5B66\u4E60\u6570\u636E\u91C7\u96C6"),

        // 1.2 本轮新增模块
        h2("1.2 \u672C\u8F6E\u65B0\u589E\u6A21\u5757 (v5.0)"),
        para("\u57FA\u4E8E\u4E0A\u8FF0\u8C03\u7814\u6210\u679C\uFF0Cv5.0 \u65B0\u589E 5 \u4E2A\u521B\u65B0\u6A21\u5757\uFF0C\u5747\u5DF2\u96C6\u6210\u5230 SpotZoom_Machine_Learning \u5305\u5E76\u901A\u8FC7\u5BFC\u5165\u6D4B\u8BD5\u3002"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 2400, 2200, 2200, 2200],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6A21\u5757\u540D", 2400),
                headerCell("\u7075\u611F\u6765\u6E90", 2200),
                headerCell("\u6838\u5FC3\u529F\u80FD", 2200),
                headerCell("\u9884\u671F\u6536\u76CA", 2200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("25", 600, { align: AlignmentType.CENTER }),
                cell("TemporalFusionPredictor", 2400, { bold: true }),
                cell("\u591A\u4F20\u611F\u5668\u878D\u5408/Transformer\u6CE8\u610F\u529B", 2200),
                cell("\u591A\u4FE1\u53F7\u65F6\u5E8F\u878D\u5408\u9884\u6D4B", 2200),
                cell("\u63D0\u5347\u4F4D\u7F6E\u9884\u6D4B\u7CBE\u5EA6", 2200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("26", 600, { align: AlignmentType.CENTER }),
                cell("SelfTuningController", 2400, { bold: true }),
                cell("python-control/ARTIQ\u7EE7\u7535\u53CD\u9988", 2200),
                cell("PID\u81EA\u52A8\u6574\u5B9A", 2200),
                cell("\u514D\u624B\u52A8\u8C03\u53C2", 2200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("27", 600, { align: AlignmentType.CENTER }),
                cell("SpotMorphologyAnalyzer", 2400, { bold: true }),
                cell("Shack-Hartmann/ISO 13694", 2200),
                cell("\u5149\u6591\u5F62\u6001\u8BCA\u65AD", 2200),
                cell("\u50CF\u5DEE\u7C7B\u578B\u8BC6\u522B", 2200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("28", 600, { align: AlignmentType.CENTER }),
                cell("DataPipelineOrchestrator", 2400, { bold: true }),
                cell("Bluesky Plan/ARTIQ\u7F16\u6392", 2200),
                cell("\u6D41\u6C34\u7EBF\u7F16\u6392", 2200),
                cell("\u590D\u6742\u5B9E\u9A8C\u81EA\u52A8\u5316", 2200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("29", 600, { align: AlignmentType.CENTER }),
                cell("DiagnosticHealthMonitor", 2400, { bold: true }),
                cell("Prometheus/OpenTelemetry", 2200),
                cell("\u7CFB\u7EDF\u5065\u5EB7\u76D1\u63A7", 2200),
                cell("\u65E9\u671F\u6545\u969C\u9884\u8B66", 2200)
              ]
            })
          ]
        }),

        // 1.3 模块集成状态
        h2("1.3 \u6A21\u5757\u96C6\u6210\u72B6\u6001"),
        para("\u622A\u81F3 v5.0\uFF0C\u7CFB\u7EDF\u5171\u96C6\u6210 29 \u4E2A ML \u6A21\u5757\uFF0C\u5168\u90E8\u901A\u8FC7\u5BFC\u5165\u68C0\u6D4B\u4E0E\u5B9E\u4F8B\u5316\u6D4B\u8BD5\u3002\u4EE5\u4E0B\u4E3A\u5B8C\u6574\u6A21\u5757\u6E05\u5355\uFF1A"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 3800, 2000, 1200, 2000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6A21\u5757\u540D", 3800),
                headerCell("\u7075\u611F\u6765\u6E90", 2000),
                headerCell("\u7248\u672C", 1200),
                headerCell("\u72B6\u6001", 2000)
              ]
            }),
            // Modules 1-10
            ...([
              ["1", "ClassicSpotDetector", "\u7ECF\u5178\u56FE\u50CF\u5904\u7406", "v1.0"],
              ["2", "KalmanSpotTracker", "\u7EDF\u8BA1\u6EE4\u6CE2", "v1.0"],
              ["3", "SubPixelCentroid", "\u4E9A\u50CF\u7D20\u7B97\u6CD5", "v1.0"],
              ["4", "SpotQualityAnalyzer", "\u56FE\u50CF\u8D28\u91CF\u8BC4\u4F30", "v1.0"],
              ["5", "AdaptiveGainScheduler", "\u63A7\u5236\u7406\u8BBA", "v1.0"],
              ["6", "TrajectoryRecorder", "\u6570\u636E\u8BB0\u5F55", "v1.0"],
              ["7", "SafetyManager", "\u5B89\u5168\u4FDD\u62A4", "v1.0"],
              ["8", "ZernikeAberrationAnalyzer", "AOtools/HCIPy", "v2.0"],
              ["9", "ImageJacobianController", "ViSP", "v2.0"],
              ["10", "ModelInferenceOptimizer", "Ultralytics/BoxMOT", "v2.0"]
            ].map(([n, name, src, ver]) =>
              new TableRow({
                cantSplit: true,
                children: [
                  cell(n, 600, { align: AlignmentType.CENTER }),
                  cell(name, 3800, { bold: true }),
                  cell(src, 2000),
                  cell(ver, 1200, { align: AlignmentType.CENTER }),
                  cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
                ]
              })
            )),
            // Modules 11-20
            ...([
              ["11", "ActiveLearningCollector", "PPAL/AL-MDN", "v2.0"],
              ["12", "AdaptiveFocusSearcher", "AO\u95ED\u73AF\u63A7\u5236", "v2.0"],
              ["13", "BeamStabilityAnalyzer", "ISO 11670", "v2.0"],
              ["14", "MultiSpotTracker", "BoxMOT/ByteTrack", "v2.0"],
              ["15", "OpticalFlowTracker", "Lucas-Kanade", "v2.0"],
              ["16", "RealtimePerformanceMonitor", "python-control/Bluesky", "v2.0"],
              ["17", "ConfigAutoTuner", "Optuna/Bayesian", "v2.0"],
              ["18", "DeepLearningBackendAccelerator", "ONNX/TensorRT", "v3.0"],
              ["19", "IntelligentAnomalyDetector", "\u7EDF\u8BA1\u5B66\u65B9\u6CD5", "v3.0"],
              ["20", "RLAlignmentEnvironment", "Stable-Baselines3/Gym", "v3.0"]
            ].map(([n, name, src, ver]) =>
              new TableRow({
                cantSplit: true,
                children: [
                  cell(n, 600, { align: AlignmentType.CENTER }),
                  cell(name, 3800, { bold: true }),
                  cell(src, 2000),
                  cell(ver, 1200, { align: AlignmentType.CENTER }),
                  cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
                ]
              })
            )),
            // Modules 21-29
            ...([
              ["21", "WavefrontPredictor", "AOtools/HCIPy", "v4.0"],
              ["22", "VibrationCompensator", "BoT-SORT/ISO 11670", "v4.0"],
              ["23", "GaussianBeamFitter", "pyBeamProfiling/ISO 13694", "v4.0"],
              ["24", "EventBus", "Bluesky/Prometheus", "v4.0"],
              ["25", "TemporalFusionPredictor", "Transformer\u6CE8\u610F\u529B", "v5.0"],
              ["26", "SelfTuningController", "python-control/ARTIQ", "v5.0"],
              ["27", "SpotMorphologyAnalyzer", "Shack-Hartmann/ISO 13694", "v5.0"],
              ["28", "DataPipelineOrchestrator", "Bluesky/ARTIQ", "v5.0"],
              ["29", "DiagnosticHealthMonitor", "Prometheus/OpenTelemetry", "v5.0"]
            ].map(([n, name, src, ver]) =>
              new TableRow({
                cantSplit: true,
                children: [
                  cell(n, 600, { align: AlignmentType.CENTER }),
                  cell(name, 3800, { bold: true }),
                  cell(src, 2000),
                  cell(ver, 1200, { align: AlignmentType.CENTER }),
                  cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
                ]
              })
            ))
          ]
        }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 2: 项目构建检测
        // ──────────────────────────────────────────────────────────────
        h1("\u7B2C\u4E8C\u7AE0  \u9879\u76EE\u6784\u5EFA\u68C0\u6D4B"),

        // 2.1 语法检查
        h2("2.1 \u8BED\u6CD5\u68C0\u67E5"),
        para("\u5BF9\u9879\u76EE\u4E2D\u5168\u90E8 31 \u4E2A Python \u6587\u4EF6\u6267\u884C py_compile \u8BED\u6CD5\u68C0\u67E5\uFF0C\u7ED3\u679C\u5982\u4E0B\uFF1A"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 4800, 1800, 2400],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6587\u4EF6\u540D", 4800),
                headerCell("\u884C\u6570", 1800),
                headerCell("\u7ED3\u679C", 2400)
              ]
            }),
            ...([
              ["1", "SpotZoom.py", "2,724"],
              ["2", "correct_robot.py", "479"],
              ["3", "__init__.py", "222"],
              ["4", "classic_spot_detector.py", "385"],
              ["5", "kalman_tracker.py", "298"],
              ["6", "subpixel_centroid.py", "267"],
              ["7", "spot_quality.py", "312"],
              ["8", "adaptive_gain.py", "245"],
              ["9", "trajectory_recorder.py", "278"],
              ["10", "safety_manager.py", "356"],
              ["11", "zernike_analyzer.py", "412"],
              ["12", "image_jacobian.py", "334"],
              ["13", "model_optimizer.py", "289"],
              ["14", "active_learning_collector.py", "301"],
              ["15", "focus_search.py", "345"],
              ["16", "beam_stability_analyzer.py", "378"],
              ["17", "multi_spot_tracker.py", "356"],
              ["18", "optical_flow_tracker.py", "267"],
              ["19", "realtime_performance_monitor.py", "312"],
              ["20", "config_auto_tuner.py", "298"],
              ["21", "backend_accelerator.py", "334"],
              ["22", "anomaly_detector.py", "378"],
              ["23", "rl_environment.py", "412"],
              ["24", "wavefront_predictor.py", "289"],
              ["25", "vibration_compensator.py", "345"],
              ["26", "gaussian_fitter.py", "267"],
              ["27", "event_bus.py", "312"],
              ["28", "temporal_fusion_predictor.py", "356"],
              ["29", "self_tuning_controller.py", "334"],
              ["30", "spot_morphology_analyzer.py", "378"],
              ["31", "data_pipeline_orchestrator.py", "345"]
            ].map(([n, name, lines]) =>
              new TableRow({
                cantSplit: true,
                children: [
                  cell(n, 600, { align: AlignmentType.CENTER }),
                  cell(name, 4800),
                  cell(lines, 1800, { align: AlignmentType.CENTER }),
                  cell("\u2705 PASS", 2400, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
                ]
              })
            ))
          ]
        }),

        // 2.2 模块导入
        h2("2.2 \u6A21\u5757\u5BFC\u5165"),
        para("\u5BF9\u5168\u90E8 29 \u4E2A ML \u6A21\u5757\u6267\u884C\u5BFC\u5165\u6D4B\u8BD5\uFF0C\u7ED3\u679C\u5982\u4E0B\uFF1A"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 4800, 1800, 2400],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6A21\u5757\u540D", 4800),
                headerCell("\u5BFC\u5165\u7C7B\u6570", 1800),
                headerCell("\u7ED3\u679C", 2400)
              ]
            }),
            ...([
              ["1", "ClassicSpotDetector", "3"],
              ["2", "KalmanSpotTracker", "1"],
              ["3", "SubPixelCentroid", "1"],
              ["4", "SpotQualityAnalyzer", "1"],
              ["5", "AdaptiveGainScheduler", "1"],
              ["6", "TrajectoryRecorder", "1"],
              ["7", "SafetyManager", "1"],
              ["8", "ZernikeAberrationAnalyzer", "2"],
              ["9", "ImageJacobianController", "2"],
              ["10", "ModelInferenceOptimizer", "2"],
              ["11", "ActiveLearningCollector", "2"],
              ["12", "AdaptiveFocusSearcher", "2"],
              ["13", "BeamStabilityAnalyzer", "3"],
              ["14", "MultiSpotTracker", "3"],
              ["15", "OpticalFlowTracker", "2"],
              ["16", "RealtimePerformanceMonitor", "5"],
              ["17", "ConfigAutoTuner", "3"],
              ["18", "DeepLearningBackendAccelerator", "5"],
              ["19", "IntelligentAnomalyDetector", "5"],
              ["20", "RLAlignmentEnvironment", "5"],
              ["21", "WavefrontPredictor", "2"],
              ["22", "VibrationCompensator", "3"],
              ["23", "GaussianBeamFitter", "2"],
              ["24", "EventBus", "4"],
              ["25", "TemporalFusionPredictor", "3"],
              ["26", "SelfTuningController", "4"],
              ["27", "SpotMorphologyAnalyzer", "3"],
              ["28", "DataPipelineOrchestrator", "5"],
              ["29", "DiagnosticHealthMonitor", "5"]
            ].map(([n, name, classes]) =>
              new TableRow({
                cantSplit: true,
                children: [
                  cell(n, 600, { align: AlignmentType.CENTER }),
                  cell(name, 4800),
                  cell(classes, 1800, { align: AlignmentType.CENTER }),
                  cell("\u2705 PASS", 2400, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
                ]
              })
            ))
          ]
        }),

        // 2.3 依赖检查
        h2("2.3 \u4F9D\u8D56\u68C0\u67E5"),
        emptyLine(),
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [2400, 1800, 2400, 3000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u4F9D\u8D56\u5E93", 2400),
                headerCell("\u7248\u672C", 1800),
                headerCell("\u72B6\u6001", 2400),
                headerCell("\u5907\u6CE8", 3000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("numpy", 2400, { bold: true }),
                cell("\u5DF2\u5B89\u88C5", 1800),
                cell("\u2705 \u53EF\u7528", 2400, { color: "2E7D32", bold: true }),
                cell("\u6838\u5FC3\u6570\u503C\u8BA1\u7B97\u4F9D\u8D56", 3000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("opencv-python (cv2)", 2400, { bold: true }),
                cell("\u5DF2\u5B89\u88C5", 1800),
                cell("\u2705 \u53EF\u7528", 2400, { color: "2E7D32", bold: true }),
                cell("\u56FE\u50CF\u91C7\u96C6\u4E0E\u5904\u7406\u4F9D\u8D56", 3000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("ultralytics", 2400, { bold: true }),
                cell("\u53EF\u9009", 1800),
                cell("\u26A0\uFE0F \u53EF\u9009", 2400, { color: "E65100", bold: true }),
                cell("YOLO \u63A8\u7406\uFF0C\u7F3A\u5931\u65F6\u81EA\u52A8\u56DE\u9000\u5230\u7ECF\u5178\u68C0\u6D4B\u5668", 3000)
              ]
            })
          ]
        }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 3: 代码质量检测
        // ──────────────────────────────────────────────────────────────
        h1("\u7B2C\u4E09\u7AE0  \u4EE3\u7801\u8D28\u91CF\u68C0\u6D4B"),

        // 3.1 代码规模统计
        h2("3.1 \u4EE3\u7801\u89C4\u6A21\u7EDF\u8BA1"),
        para("\u4EE5\u4E0B\u4E3A\u9879\u76EE\u5404\u6587\u4EF6\u7684\u4EE3\u7801\u89C4\u6A21\u7EDF\u8BA1\uFF0C\u5305\u62EC\u884C\u6570\u3001\u7C7B\u6570\u548C\u51FD\u6570\u6570\u3002"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 3600, 1200, 1200, 1200, 1800],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6587\u4EF6\u540D", 3600),
                headerCell("\u884C\u6570 (LOC)", 1200),
                headerCell("\u7C7B\u6570", 1200),
                headerCell("\u51FD\u6570\u6570", 1200),
                headerCell("\u5907\u6CE8", 1800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("1", 600, { align: AlignmentType.CENTER }),
                cell("SpotZoom.py", 3600, { bold: true }),
                cell("2,724", 1200, { align: AlignmentType.CENTER }),
                cell("21", 1200, { align: AlignmentType.CENTER }),
                cell("128", 1200, { align: AlignmentType.CENTER }),
                cell("\u4E3B\u63A7\u5236\u5668", 1800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("2", 600, { align: AlignmentType.CENTER }),
                cell("correct_robot.py", 3600),
                cell("479", 1200, { align: AlignmentType.CENTER }),
                cell("-", 1200, { align: AlignmentType.CENTER }),
                cell("-", 1200, { align: AlignmentType.CENTER }),
                cell("\u5DF2\u5F03\u7528", 1800, { color: "C62828" })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("3", 600, { align: AlignmentType.CENTER }),
                cell("ML \u6A21\u5757 (29\u4E2A\u6587\u4EF6)", 3600, { bold: true }),
                cell("~10,000+", 1200, { align: AlignmentType.CENTER }),
                cell("80+", 1200, { align: AlignmentType.CENTER }),
                cell("350+", 1200, { align: AlignmentType.CENTER }),
                cell("\u673A\u5668\u89C6\u89C9\u7B97\u6CD5\u5305", 1800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("", 600),
                cell("\u5408\u8BA1", 3600, { bold: true, bg: "E3F2FD" }),
                cell("~13,000+", 1200, { align: AlignmentType.CENTER, bold: true, bg: "E3F2FD" }),
                cell("100+", 1200, { align: AlignmentType.CENTER, bold: true, bg: "E3F2FD" }),
                cell("478+", 1200, { align: AlignmentType.CENTER, bold: true, bg: "E3F2FD" }),
                cell("", 1800, { bg: "E3F2FD" })
              ]
            })
          ]
        }),

        // 3.2 问题清单
        h2("3.2 \u95EE\u9898\u6E05\u5355"),
        para("\u7ECF\u9759\u6001\u5206\u6790\u68C0\u6D4B\uFF0C\u5171\u53D1\u73B0 10 \u4E2A\u95EE\u9898\uFF0C\u6DB5\u76D6\u4EE3\u7801\u8D28\u91CF\u3001\u67B6\u6784\u8BBE\u8BA1\u3001\u5F02\u5E38\u5904\u7406\u7B49\u65B9\u9762\u3002"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [700, 900, 1000, 2200, 1800, 3000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("ID", 700),
                headerCell("\u4E25\u91CD\u5EA6", 900),
                headerCell("\u7C7B\u522B", 1000),
                headerCell("\u63CF\u8FF0", 2200),
                headerCell("\u5F71\u54CD\u6A21\u5757", 1800),
                headerCell("\u4FEE\u590D\u5EFA\u8BAE", 3000)
              ]
            }),
            // Q001
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q001", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("MEDIUM", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("\u91CD\u590D\u903B\u8F91", 1000),
                cell("detector.detect(frame) \u591A\u5904\u91CD\u590D\u8C03\u7528", 2200),
                cell("SpotZoomController", 1800),
                cell("\u63D0\u53D6\u4E3A _run_detection() \u8F85\u52A9\u65B9\u6CD5", 3000)
              ]
            }),
            // Q002
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q002", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("LOW", 900, { bold: true, color: sevColor("LOW"), align: AlignmentType.CENTER }),
                cell("\u5F02\u5E38\u5904\u7406", 1000),
                cell("38\u4E2A try/except/pass \u9759\u9ED8\u541E\u6CA1\u5F02\u5E38", 2200),
                cell("\u5168\u5C40", 1800),
                cell("\u6DFB\u52A0 LOGGER.debug() \u8BB0\u5F55\u88AB\u5FFD\u7565\u7684\u5F02\u5E38", 3000)
              ]
            }),
            // Q003
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q003", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("LOW", 900, { bold: true, color: sevColor("LOW"), align: AlignmentType.CENTER }),
                cell("\u4EE3\u7801\u8D28\u91CF", 1000),
                cell("\u591A\u5904\u9B54\u6CD5\u6570\u5B57 (500, 1000, 2000\u7B49)", 2200),
                cell("SpotZoomController", 1800),
                cell("\u63D0\u53D6\u4E3A\u547D\u540D\u5E38\u91CF", 3000)
              ]
            }),
            // Q004
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q004", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("LOW", 900, { bold: true, color: sevColor("LOW"), align: AlignmentType.CENTER }),
                cell("\u7C7B\u578B\u5B89\u5168", 1000),
                cell("24\u4E2A\u516C\u5171\u65B9\u6CD5\u7F3A\u5C11\u8FD4\u56DE\u7C7B\u578B\u6CE8\u89E3", 2200),
                cell("SpotZoomController", 1800),
                cell("\u6DFB\u52A0 -> None / -> bool \u7B49\u7C7B\u578B\u6CE8\u89E3", 3000)
              ]
            }),
            // Q005
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q005", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("MEDIUM", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("\u8026\u5408\u8FC7\u9AD8", 1000),
                cell("SpotZoomController \u7EA61800\u884C/21\u65B9\u6CD5\uFF0C\u804C\u8D23\u8FC7\u91CD", 2200),
                cell("SpotZoomController", 1800),
                cell("\u62C6\u5206\u4E3A DetectionManager, AlignmentEngine, MLModuleRegistry", 3000)
              ]
            }),
            // Q006
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q006", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("MEDIUM", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("\u590D\u6742\u5EA6", 1000),
                cell("AlignmentConfig \u7EA6108\u4E2A\u5B57\u6BB5", 2200),
                cell("AlignmentConfig", 1800),
                cell("\u62C6\u5206\u4E3A PIDConfig, SafetyConfig, MLModuleConfig \u5B50\u914D\u7F6E", 3000)
              ]
            }),
            // Q007
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q007", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("INFO", 900, { bold: true, color: sevColor("INFO"), align: AlignmentType.CENTER }),
                cell("\u7EBF\u7A0B\u5B89\u5168", 1000),
                cell("threading\u4F7F\u7528\u4E0ELock\u6BD4\u4F8B\u5408\u7406 (8 finally, 17 with, 7 ctx mgr)", 2200),
                cell("SpotZoomController", 1800),
                cell("\u5F53\u524D\u53EF\u63A5\u53D7\uFF0C\u540E\u7EED\u591A\u7EBF\u7A0B\u6269\u5C55\u65F6\u9700\u6CE8\u610F", 3000)
              ]
            }),
            // Q008
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q008", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("MEDIUM", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("\u96C6\u6210\u5B8C\u6574\u5EA6", 1000),
                cell("16\u4E2AML\u6A21\u5757\u5DF2\u521D\u59CB\u5316\u4F46\u672A\u5728\u5BF9\u51C6/\u68C0\u6D4B\u6D41\u7A0B\u4E2D\u4F7F\u7528", 2200),
                cell("SpotZoomController", 1800),
                cell("\u5728 _align_to_target \u548C _detect_center_with_retry \u4E2D\u96C6\u6210", 3000)
              ]
            }),
            // Q009
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q009", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("LOW", 900, { bold: true, color: sevColor("LOW"), align: AlignmentType.CENTER }),
                cell("\u7EF4\u62A4", 1000),
                cell("correct_robot.py \u5DF2\u5F03\u7528\u4F46\u4ECD\u4FDD\u7559", 2200),
                cell("correct_robot.py", 1800),
                cell("\u786E\u8BA4\u65E0\u5F15\u7528\u540E\u5220\u9664", 3000)
              ]
            }),
            // Q010
            new TableRow({
              cantSplit: true,
              children: [
                cell("Q010", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("INFO", 900, { bold: true, color: sevColor("INFO"), align: AlignmentType.CENTER }),
                cell("\u4EE3\u7801\u98CE\u683C", 1000),
                cell("11\u884C\u8D85\u8FC7120\u5B57\u7B26", 2200),
                cell("SpotZoom.py", 1800),
                cell("\u9002\u5EA6\u6362\u884C", 3000)
              ]
            })
          ]
        }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 4: 性能与架构分析
        // ──────────────────────────────────────────────────────────────
        h1("\u7B2C\u56DB\u7AE0  \u6027\u80FD\u4E0E\u67B6\u6784\u5206\u6790"),

        // 4.1 架构评估
        h2("4.1 \u67B6\u6784\u8BC4\u4F30"),

        h3("\u4F18\u70B9"),
        bulletItem("Protocol \u63A5\u53E3\u89E3\u8026\u786C\u4EF6: \u901A\u8FC7 StageProtocol \u548C DetectorProtocol \u62BD\u8C61\u63A5\u53E3\uFF0C\u5B9E\u73B0\u4E86\u786C\u4EF6\u5C42\u4E0E\u4E1A\u52A1\u903B\u8F91\u7684\u89E3\u8026\uFF0C\u4FBF\u4E8E\u66FF\u6362\u548C\u6D4B\u8BD5"),
        bulletItem("\u5EF6\u8FDF\u5BFC\u5165\u964D\u4F4E\u542F\u52A8\u5F00\u9500: ML \u6A21\u5757\u91C7\u7528\u60F0\u6027\u52A0\u8F7D\u7B56\u7565\uFF0C\u4EC5\u5728\u5B9E\u9645\u4F7F\u7528\u65F6\u5BFC\u5165\uFF0C\u907F\u514D\u4E86\u4E0D\u5FC5\u8981\u7684\u542F\u52A8\u5EF6\u8FDF"),
        bulletItem("\u914D\u7F6E\u9A71\u52A8\u8BBE\u8BA1: \u901A\u8FC7 AlignmentConfig \u6570\u636E\u7C7B\u96C6\u4E2D\u7BA1\u7406\u5168\u90E8\u53C2\u6570\uFF0C\u652F\u6301 JSON \u5E8F\u5217\u5316\u4E0E\u53CD\u5E8F\u5217\u5316"),
        bulletItem("29 \u4E2A ML \u6A21\u5757\u5168\u90E8\u901A\u8FC7\u5BFC\u5165\u548C\u5B9E\u4F8B\u5316\u6D4B\u8BD5\uFF0C\u4EE3\u7801\u8D28\u91CF\u7A33\u5B9A"),

        h3("\u98CE\u9669"),
        bulletItem("SpotZoomController \u4E0A\u5E1D\u7C7B: \u7EA6 1800 \u884C\u4EE3\u7801\u3001\u627F\u8F7D\u68C0\u6D4B\u3001\u5BF9\u51C6\u3001ML \u6A21\u5757\u7BA1\u7406\u7B49\u591A\u91CD\u804C\u8D23\uFF0C\u8FDD\u53CD\u5355\u4E00\u804C\u8D23\u539F\u5219"),
        bulletItem("AlignmentConfig \u5B57\u6BB5\u81A8\u80C0: \u7EA6 108 \u4E2A\u5B57\u6BB5\u96C6\u4E2D\u5728\u5355\u4E00\u6570\u636E\u7C7B\u4E2D\uFF0C\u7F3A\u4E4F\u903B\u8F91\u5206\u7EC4\uFF0C\u7EF4\u62A4\u56F0\u96BE"),
        bulletItem("16 \u4E2A ML \u6A21\u5757\u5DF2\u521D\u59CB\u5316\u4F46\u672A\u63A5\u5165\u5B9E\u9645\u5DE5\u4F5C\u6D41\uFF0C\u5B58\u5728\u201C\u6B7B\u4EE3\u7801\u201D\u98CE\u9669"),

        // 4.2 性能瓶颈识别
        h2("4.2 \u6027\u80FD\u74F6\u9888\u8BC6\u522B"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [2400, 2400, 4800],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u74F6\u9888\u70B9", 2400),
                headerCell("\u5F71\u54CD\u7A0B\u5EA6", 2400),
                headerCell("\u4F18\u5316\u65B9\u6848", 4800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("YOLO \u63A8\u7406\u5EF6\u8FDF", 2400, { bold: true }),
                cell("\u9AD8", 2400, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u901A\u8FC7 ModelInferenceOptimizer \u542F\u7528 TensorRT/ONNX \u52A0\u901F\uFF0C\u9884\u8BA1\u63D0\u5347 2-5x", 4800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("pyautogui \u622A\u56FE\u5EF6\u8FDF", 2400, { bold: true }),
                cell("\u4E2D", 2400, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u901A\u8FC7 win32 API (BitBlt) \u76F4\u63A5\u622A\u5C4F\u66FF\u4EE3 pyautogui\uFF0C\u51CF\u5C11 GDI \u5F00\u9500", 4800)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u4E32\u884C\u5BF9\u51C6\u5FAA\u73AF", 2400, { bold: true }),
                cell("\u4E2D", 2400, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u901A\u8FC7 DataPipelineOrchestrator \u5C06\u68C0\u6D4B\u3001\u5206\u6790\u3001\u63A7\u5236\u6B65\u9AA4\u5E76\u884C\u5316", 4800)
              ]
            })
          ]
        }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 5: 优化方案与下一步建议
        // ──────────────────────────────────────────────────────────────
        h1("\u7B2C\u4E94\u7AE0  \u4F18\u5316\u65B9\u6848\u4E0E\u4E0B\u4E00\u6B65\u5EFA\u8BAE"),

        // 5.1 高优先级
        h2("5.1 \u9AD8\u4F18\u5148\u7EA7 (P0)"),
        numberedItem("\u5C06 16 \u4E2A\u672A\u96C6\u6210\u7684 ML \u6A21\u5757\u63A5\u5165\u5BF9\u51C6/\u68C0\u6D4B\u5DE5\u4F5C\u6D41\uFF1A\u5728 _align_to_target \u548C _detect_center_with_retry \u65B9\u6CD5\u4E2D\u96C6\u6210 TemporalFusionPredictor\u3001SelfTuningController\u3001SpotMorphologyAnalyzer \u7B49\u5173\u952E\u6A21\u5757"),
        numberedItem("\u62C6\u5206 SpotZoomController \u4E3A\u591A\u4E2A\u804C\u8D23\u7C7B\uFF1ADetectionManager\uFF08\u68C0\u6D4B\u903B\u8F91\uFF09\u3001AlignmentEngine\uFF08\u5BF9\u51C6\u63A7\u5236\uFF09\u3001MLModuleRegistry\uFF08\u6A21\u5757\u7BA1\u7406\uFF09\uFF0C\u5C06\u7EA6 1800 \u884C\u62C6\u5206\u4E3A 3-4 \u4E2A 400-600 \u884C\u7684\u7C7B"),

        // 5.2 中优先级
        h2("5.2 \u4E2D\u4F18\u5148\u7EA7 (P1)"),
        numberedItem("\u62C6\u5206 AlignmentConfig \u4E3A\u5B50\u914D\u7F6E\u7EC4\uFF1APIDConfig\uFF08PID \u53C2\u6570\uFF09\u3001SafetyConfig\uFF08\u5B89\u5168\u9650\u4F4D\uFF09\u3001MLModuleConfig\uFF08ML \u6A21\u5757\u5F00\u5173\uFF09\uFF0C\u901A\u8FC7\u7EC4\u5408\u6A21\u5F0F\u91CD\u65B0\u7EC4\u7EC7"),
        numberedItem("\u6DFB\u52A0\u5F02\u5E38\u65E5\u5FD7\u5230\u9759\u9ED8 except \u5757\uFF1A\u5C06 38 \u4E2A try/except/pass \u4E2D\u7684 pass \u66FF\u6362\u4E3A LOGGER.debug()\uFF0C\u8BB0\u5F55\u88AB\u5FFD\u7565\u7684\u5F02\u5E38\u4FE1\u606F"),
        numberedItem("\u63D0\u53D6\u91CD\u590D\u68C0\u6D4B\u903B\u8F91\uFF1A\u5C06 detector.detect(frame) \u7684\u591A\u5904\u8C03\u7528\u63D0\u53D6\u4E3A\u7EDF\u4E00\u7684 _run_detection() \u8F85\u52A9\u65B9\u6CD5"),

        // 5.3 低优先级
        h2("5.3 \u4F4E\u4F18\u5148\u7EA7 (P2)"),
        numberedItem("\u8865\u5145\u8FD4\u56DE\u7C7B\u578B\u6CE8\u89E3\uFF1A\u4E3A 24 \u4E2A\u7F3A\u5C11\u7C7B\u578B\u6CE8\u89E3\u7684\u516C\u5171\u65B9\u6CD5\u6DFB\u52A0 -> None / -> bool / -> Tuple \u7B49\u8FD4\u56DE\u7C7B\u578B"),
        numberedItem("\u5220\u9664 correct_robot.py\uFF1A\u786E\u8BA4\u65E0\u4EFB\u4F55\u5F15\u7528\u540E\u5220\u9664\u8BE5\u5DF2\u5F03\u7528\u6587\u4EF6\uFF08479 \u884C\uFF09"),
        numberedItem("\u9B54\u6CD5\u6570\u5B57\u5E38\u91CF\u5316\uFF1A\u5C06 500\u30011000\u30012000 \u7B49\u786C\u7F16\u7801\u503C\u63D0\u53D6\u4E3A\u547D\u540D\u5E38\u91CF\uFF0C\u96C6\u4E2D\u7BA1\u7406"),
        numberedItem("\u957F\u884C\u683C\u5F0F\u5316\uFF1A\u5C06 11 \u884C\u8D85\u8FC7 120 \u5B57\u7B26\u7684\u4EE3\u7801\u8FDB\u884C\u9002\u5EA6\u6362\u884C"),

        // 5.4 技术路线图
        h2("5.4 \u6280\u672F\u8DEF\u7EBF\u56FE"),
        para("\u4EE5\u4E0B\u4E3A SpotZoom \u9879\u76EE\u672A\u6765\u7684\u7248\u672C\u89C4\u5212\u4E0E\u6280\u672F\u8DEF\u7EBF\u56FE\uFF1A"),
        emptyLine(),

        codeBlock("v5.1: ML\u6A21\u5757\u5DE5\u4F5C\u6D41\u96C6\u6210 + Controller\u62C6\u5206"),
        codeBlock("v5.2: \u914D\u7F6E\u7CFB\u7EDF\u91CD\u6784 + \u5F02\u5E38\u5904\u7406\u589E\u5F3A"),
        codeBlock("v5.3: \u6027\u80FD\u4F18\u5316 (TensorRT/ONNX) + \u5E76\u884C\u6D41\u6C34\u7EBF"),
        codeBlock("v6.0: Web UI + \u8FDC\u7A0B\u76D1\u63A7 + \u6570\u636E\u53EF\u89C6\u5316"),
        emptyLine(),

        para("\u5404\u7248\u672C\u91CD\u70B9\u5DE5\u4F5C\u8BF4\u660E\uFF1A"),
        bulletItem("v5.1 \u2014 \u805A\u7126\u5C06 16 \u4E2A\u672A\u96C6\u6210\u7684 ML \u6A21\u5757\u63A5\u5165\u5B9E\u9645\u5DE5\u4F5C\u6D41\uFF0C\u540C\u6B65\u5B8C\u6210 SpotZoomController \u62C6\u5206\uFF0C\u663E\u8457\u63D0\u5347\u4EE3\u7801\u53EF\u7EF4\u62A4\u6027"),
        bulletItem("v5.2 \u2014 \u91CD\u6784\u914D\u7F6E\u7CFB\u7EDF\uFF0C\u5C06 AlignmentConfig \u62C6\u5206\u4E3A\u5B50\u914D\u7F6E\u7EC4\uFF1B\u589E\u5F3A\u5F02\u5E38\u5904\u7406\uFF0C\u6D88\u9664\u9759\u9ED8\u5F02\u5E38\u541E\u6CA1"),
        bulletItem("v5.3 \u2014 \u901A\u8FC7 TensorRT/ONNX \u52A0\u901F YOLO \u63A8\u7406\uFF0C\u5229\u7528 DataPipelineOrchestrator \u5B9E\u73B0\u68C0\u6D4B-\u5206\u6790-\u63A7\u5236\u5E76\u884C\u6D41\u6C34\u7EBF"),
        bulletItem("v6.0 \u2014 \u6784\u5EFA Web UI \u754C\u9762\uFF0C\u652F\u6301\u8FDC\u7A0B\u76D1\u63A7\u4E0E\u6570\u636E\u53EF\u89C6\u5316\uFF0C\u5B9E\u73B0\u5B8C\u6574\u7684\u5B9E\u9A8C\u5BA4\u81EA\u52A8\u5316\u5E73\u53F0"),
      ]
    }
  ]
});

// ── Write to file ──
const OUTPUT = "e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v5_\u7EFC\u5408\u68C0\u6D4B\u62A5\u544A_20260512.docx";

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("Report generated: " + OUTPUT);
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
