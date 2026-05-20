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

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
}

// ── Severity color helper ──
function sevColor(sev) {
  switch (sev) {
    case "CRITICAL": return "B71C1C";
    case "HIGH": return "C62828";
    case "MEDIUM": return "E65100";
    case "LOW": return "F57F17";
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
            children: [new TextRun({ text: "SpotZoom v6.0 \u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", font: FONT_RUN(16), color: "999999" })]
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
          children: [new TextRun({ text: "SpotZoom v6.0 \u7EFC\u5408\u68C0\u6D4B\u62A5\u544A", bold: true, font: FONT_RUN(48), color: "1F4E79" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({
            text: "\u5149\u6591\u95ED\u73AF\u5BF9\u51C6\u7CFB\u7EDF \u2014 \u4EE3\u7801\u8D28\u91CF\u3001\u67B6\u6784\u4E0E\u6A21\u5757\u5065\u5EB7\u5EA6\u5206\u6790",
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
          children: [new TextRun({ text: "\u9879\u76EE: 8821L", font: FONT_RUN(22), color: "666666" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "\u5206\u6790\u8303\u56F4: 33 \u4E2A\u6587\u4EF6 / 17,126 \u884C\u4EE3\u7801 / 528 \u4E2A\u51FD\u6570 / 119 \u4E2A\u7C7B", font: FONT_RUN(20), color: "888888" })]
        }),
        new Paragraph({ children: [new PageBreak()] }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 1: 执行摘要
        // ──────────────────────────────────────────────────────────────
        h1("\u4E00\u3001\u6267\u884C\u6458\u8981"),

        // Health score highlight
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [3000, 3000, 3600],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u6307\u6807", 3000),
                headerCell("\u6570\u503C", 3000),
                headerCell("\u8BF4\u660E", 3600)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u603B\u4F53\u5065\u5EB7\u8BC4\u5206", 3000, { bold: true }),
                cell("58.6 / 100", 3000, { bold: true, color: "E65100", align: AlignmentType.CENTER }),
                cell("\u4E2D\u7B49\u504F\u4E0B\uFF0C\u9700\u91CD\u70B9\u6539\u8FDB", 3600)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u5206\u6790\u6587\u4EF6\u6570", 3000),
                cell("33", 3000, { align: AlignmentType.CENTER }),
                cell("Python + JS + BAT", 3600)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u4EE3\u7801\u603B\u884C\u6570", 3000),
                cell("17,126", 3000, { align: AlignmentType.CENTER }),
                cell("\u542B\u6CE8\u91CA\u548C\u7A7A\u884C", 3600)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u51FD\u6570\u603B\u6570", 3000),
                cell("528", 3000, { align: AlignmentType.CENTER }),
                cell("\u542B\u7C7B\u65B9\u6CD5\u4E0E\u72EC\u7ACB\u51FD\u6570", 3600)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u7C7B\u603B\u6570", 3000),
                cell("119", 3000, { align: AlignmentType.CENTER }),
                cell("\u542B\u6570\u636E\u7C7B\u4E0E\u63A7\u5236\u5668\u7C7B", 3600)
              ]
            })
          ]
        }),

        emptyLine(),

        // Key findings
        h2("\u5173\u952E\u53D1\u73B0"),
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [2400, 1200, 6000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u4E25\u91CD\u7EA7\u522B", 2400),
                headerCell("\u6570\u91CF", 1200),
                headerCell("\u5178\u578B\u95EE\u9898", 6000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u4E25\u91CD (Critical)", 2400, { bold: true, color: sevColor("CRITICAL") }),
                cell("4", 1200, { bold: true, align: AlignmentType.CENTER, color: sevColor("CRITICAL") }),
                cell("God Class\u3001\u6587\u4EF6\u8FC7\u957F\u3001\u5BFC\u5165\u541E\u9519\u3001\u786C\u7F16\u7801\u5BC6\u7801", 6000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u9AD8\u4F18\u5148\u7EA7 (High)", 2400, { bold: true, color: sevColor("HIGH") }),
                cell("6", 1200, { bold: true, align: AlignmentType.CENTER, color: sevColor("HIGH") }),
                cell("\u9759\u9ED8except\u3001\u5708\u590D\u6742\u5EA6\u3001\u914D\u7F6E\u81A8\u80C0\u3001\u8D44\u6E90\u6CC4\u6F0F", 6000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u4E2D\u7B49 (Medium)", 2400, { bold: true, color: sevColor("MEDIUM") }),
                cell("8", 1200, { bold: true, align: AlignmentType.CENTER, color: sevColor("MEDIUM") }),
                cell("\u9B54\u6CD5\u6570\u5B57\u3001\u9664\u96F6\u98CE\u9669\u3001\u7EBF\u7A0B\u5B89\u5168\u3001\u7F3A\u5C11\u6D4B\u8BD5", 6000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("\u4F4E\u4F18\u5148\u7EA7 (Low)", 2400, { bold: true, color: sevColor("LOW") }),
                cell("4", 1200, { bold: true, align: AlignmentType.CENTER, color: sevColor("LOW") }),
                cell("\u5F03\u7528\u6587\u4EF6\u3001\u7F16\u7801\u8BBE\u7F6E\u3001\u4EE3\u7801\u91CD\u590D\u3001\u5E73\u53F0\u4F9D\u8D56", 6000)
              ]
            })
          ]
        }),

        emptyLine(),

        // Overall assessment
        h2("\u603B\u4F53\u8BC4\u4EF7"),
        para("ML\u5B50\u6A21\u5757\u8D28\u91CF\u4F18\u79C0\uFF1A30\u4E2A\u6A21\u5757\uFF0C\u7C7B\u578B\u63D0\u793A\u8986\u76D6\u7387 94.8%\uFF0C\u6587\u6863\u8986\u76D6\u7387 84.1%\uFF0C\u5168\u90E8\u901A\u8FC7\u5BFC\u5165\u68C0\u6D4B\u4E0E\u529F\u80FD\u6D4B\u8BD5\u3002\u4F46\u4E3B\u7A0B\u5E8F SpotZoom.py \u5B58\u5728\u8F83\u5927\u6280\u672F\u503A\u52A1\uFF0C\u5305\u62EC God Class\u3001\u6587\u4EF6\u8FC7\u957F\u3001\u5927\u91CF\u9759\u9ED8\u5F02\u5E38\u541E\u6CA1\u7B49\u95EE\u9898\uFF0C\u4E25\u91CD\u5F71\u54CD\u53EF\u7EF4\u62A4\u6027\u4E0E\u53EF\u9760\u6027\u3002"),
        para("\u5EFA\u8BAE\u4F18\u5148\u5904\u7406 4 \u4E2A\u4E25\u91CD\u95EE\u9898\u548C 6 \u4E2A\u9AD8\u4F18\u5148\u7EA7\u95EE\u9898\uFF0C\u9884\u8BA1\u53EF\u5C06\u5065\u5EB7\u8BC4\u5206\u63D0\u5347\u81F3 75+ \u5206\u3002"),

        // ──────────────────────────────────────────────────────────────
        // SECTION 2: 本轮新增模块
        // ──────────────────────────────────────────────────────────────
        h1("\u4E8C\u3001\u672C\u8F6E\u65B0\u589E\u6A21\u5757 (v6.0)"),
        para("\u57FA\u4E8E v5.0 \u7684\u8C03\u7814\u6210\u679C\uFF0Cv6.0 \u65B0\u589E 5 \u4E2A\u521B\u65B0\u6A21\u5757\uFF0C\u5747\u5DF2\u96C6\u6210\u5230 SpotZoom_Machine_Learning \u5305\u5E76\u901A\u8FC7\u5BFC\u5165\u6D4B\u8BD5\u4E0E\u529F\u80FD\u6D4B\u8BD5\u3002\u6240\u6709\u65B0\u6A21\u5757\u4EC5\u4F9D\u8D56 numpy \u548C cv2\uFF0C\u65E0\u989D\u5916\u4F9D\u8D56\u3002"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [600, 2400, 2400, 2200, 2000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("#", 600),
                headerCell("\u6A21\u5757\u540D", 2400),
                headerCell("\u7075\u611F\u6765\u6E90", 2400),
                headerCell("\u6838\u5FC3\u529F\u80FD", 2200),
                headerCell("\u72B6\u6001", 2000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("1", 600, { align: AlignmentType.CENTER }),
                cell("PhaseRetrievalAnalyzer", 2400, { bold: true }),
                cell("HCIPy / AOtools", 2400),
                cell("Gerchberg-Saxton \u76F8\u4F4D\u6062\u590D\u6CE2\u524D\u5206\u6790", 2200),
                cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("2", 600, { align: AlignmentType.CENTER }),
                cell("PSFEstimator", 2400, { bold: true }),
                cell("HCIPy / AOtools", 2400),
                cell("PSF \u8D28\u91CF\u5B9E\u65F6\u4F30\u8BA1", 2200),
                cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("3", 600, { align: AlignmentType.CENTER }),
                cell("ModalController", 2400, { bold: true }),
                cell("AOtools / SOAPY", 2400),
                cell("Zernike \u6A21\u6001\u63A7\u5236\u5668", 2200),
                cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("4", 600, { align: AlignmentType.CENTER }),
                cell("AtmosphericTurbulenceSimulator", 2400, { bold: true }),
                cell("HCIPy / AOtools", 2400),
                cell("\u5927\u6C14\u6E4D\u6D41\u4EFF\u771F\u5668", 2200),
                cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("5", 600, { align: AlignmentType.CENTER }),
                cell("SmartRefinementController", 2400, { bold: true }),
                cell("ViSP / python-control", 2400),
                cell("\u667A\u80FD\u7CBE\u4FEE\u63A7\u5236\u5668", 2200),
                cell("\u2705 PASS", 2000, { color: "2E7D32", bold: true, align: AlignmentType.CENTER })
              ]
            })
          ]
        }),

        emptyLine(),
        para("\u6A21\u5757\u96C6\u6210\u72B6\u6001\u603B\u7ED3\uFF1A\u5168\u90E8 5 \u4E2A\u65B0\u6A21\u5757\u5BFC\u5165\u9A8C\u8BC1\u901A\u8FC7\uFF0C\u529F\u80FD\u6D4B\u8BD5\u901A\u8FC7\uFF0C\u4EC5\u4F9D\u8D56 numpy + cv2\uFF0C\u65E0\u989D\u5916\u7B2C\u4E09\u65B9\u5E93\u4F9D\u8D56\u3002\u622A\u81F3 v6.0\uFF0CML \u6A21\u5757\u603B\u6570\u8FBE\u5230 34 \u4E2A\uFF0C\u7C7B\u578B\u63D0\u793A\u8986\u76D6\u7387 94.8%\uFF0C\u6587\u6863\u8986\u76D6\u7387 84.1%\u3002"),

        // ──────────────────────────────────────────────────────────────
        // SECTION 3: 问题检测详情
        // ──────────────────────────────────────────────────────────────
        h1("\u4E09\u3001\u95EE\u9898\u68C0\u6D4B\u8BE6\u60C5"),
        para("\u7ECF\u9759\u6001\u5206\u6790\u68C0\u6D4B\uFF0C\u5171\u53D1\u73B0 22 \u4E2A\u95EE\u9898\uFF0C\u6DB5\u76D6\u4EE3\u7801\u8D28\u91CF\u3001\u67B6\u6784\u8BBE\u8BA1\u3001\u5B89\u5168\u6027\u3001\u5F02\u5E38\u5904\u7406\u7B49\u65B9\u9762\u3002"),
        emptyLine(),

        // 3.1 Critical
        h2("3.1 \u4E25\u91CD\u95EE\u9898 (Critical, 4\u4E2A)"),

        h3("C001: SpotZoomController God Class"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1ASpotZoomController \u7C7B\u8FBE 672 \u884C\uFF0C\u627F\u8F7D\u68C0\u6D4B\u3001\u5BF9\u51C6\u3001ML\u6A21\u5757\u7BA1\u7406\u3001\u786C\u4EF6\u63A7\u5236\u7B49\u591A\u91CD\u804C\u8D23\uFF0C\u4E25\u91CD\u8FDD\u53CD\u5355\u4E00\u804C\u8D23\u539F\u5219\u3002"),
        para("\u5F71\u54CD\uFF1A\u4EE3\u7801\u96BE\u4EE5\u7406\u89E3\u3001\u6D4B\u8BD5\u56F0\u96BE\u3001\u4FEE\u6539\u98CE\u9669\u9AD8\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u62C6\u5206\u4E3A DetectionManager\u3001AlignmentEngine\u3001MLModuleRegistry \u7B49\u72EC\u7ACB\u7C7B\u3002"),

        h3("C002: SpotZoom.py \u6587\u4EF6\u8FC7\u957F"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1ASpotZoom.py \u8FBE 2723 \u884C\uFF0C\u8FDC\u8D85 500 \u884C\u5408\u7406\u9650\u5236\uFF0C\u5305\u542B\u591A\u4E2A\u7C7B\u5B9A\u4E49\u3001\u914D\u7F6E\u7C7B\u3001CLI\u53C2\u6570\u89E3\u6790\u7B49\u3002"),
        para("\u5F71\u54CD\uFF1A\u6587\u4EF6\u52A0\u8F7D\u6162\u3001\u5BFC\u822A\u56F0\u96BE\u3001\u5408\u5E76\u51B2\u7A81\u9891\u7E41\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u6309\u529F\u80FD\u62C6\u5206\u4E3A\u591A\u4E2A\u6A21\u5757\u6587\u4EF6\uFF0C\u5982 controller.py\u3001config.py\u3001cli.py \u7B49\u3002"),

        h3("C003: ML\u6A21\u5757\u5BFC\u5165\u9759\u9ED8\u541E\u9519"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1AML \u6A21\u5757\u7684\u5EF6\u8FDF\u5BFC\u5165\u4E2D\uFF0C\u5F02\u5E38\u88AB\u9759\u9ED8\u6355\u83B7\u5E76\u5FFD\u7565\uFF0C\u5BFC\u81F4\u7F16\u7A0B\u9519\u8BEF\u88AB\u9690\u85CF\uFF0C\u8C03\u8BD5\u65F6\u65E0\u6CD5\u53D1\u73B0\u95EE\u9898\u3002"),
        para("\u5F71\u54CD\uFF1A\u5BFC\u5165\u5931\u8D25\u65F6\u65E0\u4EFB\u4F55\u63D0\u793A\uFF0C\u5BFC\u81F4\u8FD0\u884C\u65F6\u5D29\u6E83\u96BE\u4EE5\u5B9A\u4F4D\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u6DFB\u52A0 logger.warning() \u8BB0\u5F55\u5BFC\u5165\u5931\u8D25\u539F\u56E0\uFF0C\u5E76\u63D0\u4F9B\u660E\u786E\u7684\u9519\u8BEF\u4FE1\u606F\u3002"),

        h3("C004: XPSZAxis \u786C\u7F16\u7801\u5BC6\u7801"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1AXPSZAxis \u7C7B\u4E2D\u786C\u7F16\u7801\u4E86\u63A7\u5236\u5668\u5BC6\u7801\uFF0C\u5B58\u5728\u5B89\u5168\u98CE\u9669\u3002"),
        para("\u5F71\u54CD\uFF1A\u5BC6\u7801\u6CC4\u9732\u98CE\u9669\uFF0C\u4E0D\u7B26\u5408\u5B89\u5168\u7F16\u7801\u89C4\u8303\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u4ECE\u73AF\u5883\u53D8\u91CF\u6216\u52A0\u5BC6\u914D\u7F6E\u6587\u4EF6\u4E2D\u8BFB\u53D6\u5BC6\u7801\uFF0C\u7981\u6B62\u6E90\u7801\u786C\u7F16\u7801\u3002"),

        emptyLine(),

        // 3.2 High
        h2("3.2 \u9AD8\u4F18\u5148\u7EA7\u95EE\u9898 (High, 6\u4E2A)"),

        h3("H005: 75\u5904\u9759\u9ED8except\u5757"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u5168\u5C40\u8303\u56F4\u5185\u53D1\u73B0 75 \u5904 try/except/pass \u9759\u9ED8\u541E\u6CA1\u5F02\u5E38\u7684\u4EE3\u7801\u5757\uFF0C\u5F02\u5E38\u4FE1\u606F\u5B8C\u5168\u4E22\u5931\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u5C06 pass \u66FF\u6362\u4E3A logger.debug() \u6216 logger.warning()\uFF0C\u8BB0\u5F55\u88AB\u5FFD\u7565\u7684\u5F02\u5E38\u3002"),

        h3("H006: 4\u4E2A\u9AD8\u5708\u590D\u6742\u5EA6\u51FD\u6570"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u53D1\u73B0 4 \u4E2A\u51FD\u6570\u5708\u590D\u6742\u5EA6\u8D85\u8FC7 15\uFF0C\u6700\u9AD8\u8FBE 64\uFF0C\u4EE3\u7801\u903B\u8F91\u96BE\u4EE5\u7406\u89E3\u548C\u6D4B\u8BD5\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u901A\u8FC7\u63D0\u53D6\u5B50\u51FD\u6570\u3001\u65E9\u8FD4\u56DE\u7B49\u7B56\u7565\u964D\u4F4E\u590D\u6742\u5EA6\u3002"),

        h3("H007: AlignmentConfig\u5B57\u6BB5\u8FC7\u591A"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1AAlignmentConfig \u6570\u636E\u7C7B\u5305\u542B 50+ \u53C2\u6570\u5B57\u6BB5\uFF0C\u7F3A\u4E4F\u903B\u8F91\u5206\u7EC4\uFF0C\u7EF4\u62A4\u56F0\u96BE\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u62C6\u5206\u4E3A PIDConfig\u3001SafetyConfig\u3001MLModuleConfig \u7B49\u5B50\u914D\u7F6E\u7C7B\u3002"),

        h3("H008: parse_args\u8FD4\u56DE\u7C7B\u578B\u4E0D\u4E00\u81F4"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1Aparse_args \u51FD\u6570\u5728\u4E0D\u540C\u5206\u652F\u8FD4\u56DE\u4E0D\u540C\u7C7B\u578B\uFF0C\u7F3A\u5C11\u7EDF\u4E00\u7684\u8FD4\u56DE\u7C7B\u578B\u6CE8\u89E3\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u7EDF\u4E00\u8FD4\u56DE\u7C7B\u578B\uFF0C\u6DFB\u52A0\u7C7B\u578B\u6CE8\u89E3\u3002"),

        h3("H009: bat\u811A\u672C\u786C\u7F16\u7801\u8DEF\u5F84"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1Arun_spotzoom.bat \u4E2D\u786C\u7F16\u7801\u4E86\u7EDD\u5BF9\u8DEF\u5F84\uFF0C\u5728\u5176\u4ED6\u673A\u5668\u4E0A\u65E0\u6CD5\u8FD0\u884C\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u4F7F\u7528\u76F8\u5BF9\u8DEF\u5F84\u6216\u73AF\u5883\u53D8\u91CF\u3002"),

        h3("H010: close()\u8D44\u6E90\u7BA1\u7406\u4E0D\u5B8C\u5584"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u90E8\u5206\u8D44\u6E90\uFF08\u76F8\u673A\u3001\u8FDE\u63A5\u7B49\uFF09\u7684 close() \u65B9\u6CD5\u7F3A\u5C11\u5B8C\u5584\u7684\u8D44\u6E90\u91CA\u653E\u903B\u8F91\uFF0C\u672A\u4F7F\u7528 context manager\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u91C7\u7528 with \u8BED\u53E5\u6216\u5B9E\u73B0 __enter__/__exit__ \u534F\u8BAE\u3002"),

        emptyLine(),

        // 3.3 Medium
        h2("3.3 \u4E2D\u7B49\u95EE\u9898 (Medium, 8\u4E2A)"),

        h3("M011: 596\u4E2A\u9B54\u6CD5\u6570\u5B57"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u5168\u5C40\u8303\u56F4\u5185\u53D1\u73B0 596 \u4E2A\u786C\u7F16\u7801\u7684\u6570\u5B57\u5E38\u91CF\uFF0C\u7F3A\u4E4F\u8BED\u4E49\u8BF4\u660E\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u63D0\u53D6\u4E3A\u547D\u540D\u5E38\u91CF\uFF0C\u96C6\u4E2D\u7BA1\u7406\u3002"),

        h3("M012: 92\u5904\u9664\u96F6\u98CE\u9669"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u53D1\u73B0 92 \u5904\u6F5C\u5728\u7684\u9664\u96F6\u64CD\u4F5C\uFF0C\u672A\u8FDB\u884C\u9632\u62A4\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u6DFB\u52A0\u96F6\u503C\u68C0\u67E5\u6216\u4F7F\u7528 numpy.errstate\u3002"),

        h3("M013: 29\u5904\u65E0\u754C\u96C6\u5408"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u53D1\u73B0 29 \u5904\u4F7F\u7528\u65E0\u754C\u96C6\u5408\uFF08\u5982 list.append \u5728\u5FAA\u73AF\u4E2D\uFF09\uFF0C\u53EF\u80FD\u5BFC\u81F4\u5185\u5B58\u6CC4\u6F0F\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u8BBE\u7F6E\u5408\u7406\u7684\u4E0A\u9650\u6216\u4F7F\u7528 deque\u3002"),

        h3("M014: \u7EBF\u7A0B\u5B89\u5168\u95EE\u9898"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u90E8\u5206\u5171\u4EAB\u72B6\u6001\u7F3A\u5C11\u7EBF\u7A0B\u540C\u6B65\u4FDD\u62A4\uFF0C\u5728\u591A\u7EBF\u7A0B\u73AF\u5883\u4E0B\u53EF\u80FD\u4EA7\u751F\u7ADE\u6001\u6761\u4EF6\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u6DFB\u52A0 Lock \u6216\u4F7F\u7528\u7EBF\u7A0B\u5B89\u5168\u7684\u6570\u636E\u7ED3\u6784\u3002"),

        h3("M015: YOLO\u6807\u6CE8\u786C\u7F16\u7801\u5C3A\u5BF8"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1AYOLO \u6A21\u578B\u7684\u6807\u6CE8\u5C3A\u5BF8\u786C\u7F16\u7801\u5728\u4EE3\u7801\u4E2D\uFF0C\u4E0D\u540C\u5206\u8FA8\u7387\u4E0B\u53EF\u80FD\u4E0D\u51C6\u786E\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u5C06\u6807\u6CE8\u5C3A\u5BF8\u63D0\u53D6\u4E3A\u914D\u7F6E\u53C2\u6570\u3002"),

        h3("M016: \u914D\u7F6E\u9A8C\u8BC1\u4E0D\u8DB3"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u914D\u7F6E\u53C2\u6570\u7F3A\u5C11\u6709\u6548\u6027\u9A8C\u8BC1\uFF0C\u975E\u6CD5\u503C\u53EF\u80FD\u5BFC\u81F4\u8FD0\u884C\u65F6\u5F02\u5E38\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u6DFB\u52A0\u914D\u7F6E\u9A8C\u8BC1\u5668\uFF0C\u5728\u542F\u52A8\u65F6\u68C0\u67E5\u53C2\u6570\u6709\u6548\u6027\u3002"),

        h3("M017: \u65E5\u5FD7\u7EA7\u522B\u4E0D\u4E00\u81F4"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u4E0D\u540C\u6A21\u5757\u4F7F\u7528\u4E0D\u540C\u7684\u65E5\u5FD7\u7EA7\u522B\u7B56\u7565\uFF0C\u7F3A\u4E4F\u7EDF\u4E00\u89C4\u8303\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u5236\u5B9A\u7EDF\u4E00\u7684\u65E5\u5FD7\u89C4\u8303\uFF0C\u4F7F\u7528\u7EDF\u4E00\u7684 logger \u914D\u7F6E\u3002"),

        h3("M018: \u7F3A\u5C11\u5355\u5143\u6D4B\u8BD5"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u9879\u76EE\u7F3A\u5C11\u5355\u5143\u6D4B\u8BD5\uFF0C\u4EE3\u7801\u53D8\u66F4\u65E0\u6CD5\u81EA\u52A8\u9A8C\u8BC1\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u4E3A\u6838\u5FC3\u6A21\u5757\u6DFB\u52A0 pytest \u5355\u5143\u6D4B\u8BD5\u3002"),

        emptyLine(),

        // 3.4 Low
        h2("3.4 \u4F4E\u4F18\u5148\u7EA7\u95EE\u9898 (Low, 4\u4E2A)"),

        h3("L019: \u5F03\u7528\u6587\u4EF6\u672A\u5220\u9664"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1Acorrect_robot.py \u5DF2\u5F03\u7528\u4F46\u4ECD\u4FDD\u7559\u5728\u9879\u76EE\u4E2D\uFF0C\u5E72\u6270\u4EE3\u7801\u5BA1\u8BA1\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u786E\u8BA4\u65E0\u5F15\u7528\u540E\u5220\u9664\u3002"),

        h3("L020: os.system\u7F16\u7801\u8BBE\u7F6E"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u4F7F\u7528 os.system \u8BBE\u7F6E\u7F16\u7801\uFF0C\u8DE8\u5E73\u53F0\u517C\u5BB9\u6027\u4E0D\u4F73\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u4F7F\u7528 locale \u6A21\u5757\u6216 subprocess \u66FF\u4EE3\u3002"),

        h3("L021: bbox\u88C1\u5207\u4EE3\u7801\u91CD\u590D"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1Abbox \u88C1\u5207\u903B\u8F91\u5728\u591A\u5904\u91CD\u590D\u5B9E\u73B0\uFF0C\u8FDD\u53CD DRY \u539F\u5219\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u63D0\u53D6\u4E3A\u516C\u5171\u5DE5\u5177\u51FD\u6570\u3002"),

        h3("L022: Windows\u5E73\u53F0\u4F9D\u8D56"),
        para("\u95EE\u9898\u63CF\u8FF0\uFF1A\u90E8\u5206\u4EE3\u7801\u4F9D\u8D56 Windows \u7279\u6709 API\uFF08\u5982 pyautogui\u3001win32gui\uFF09\uFF0C\u65E0\u6CD5\u8DE8\u5E73\u53F0\u8FD0\u884C\u3002"),
        para("\u4FEE\u590D\u5EFA\u8BAE\uFF1A\u901A\u8FC7\u62BD\u8C61\u5C42\u5C01\u88C5\u5E73\u53F0\u76F8\u5173\u8C03\u7528\u3002"),

        // ──────────────────────────────────────────────────────────────
        // SECTION 4: 模块质量评估
        // ──────────────────────────────────────────────────────────────
        h1("\u56DB\u3001\u6A21\u5757\u8D28\u91CF\u8BC4\u4F30"),
        para("\u4EE5\u4E0B\u4E3A\u5404\u6A21\u5757\u533A\u57DF\u7684\u8D28\u91CF\u6307\u6807\u8BC4\u4F30\uFF1A"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [2200, 1200, 1200, 1200, 1400, 1200],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u6A21\u5757\u533A\u57DF", 2200),
                headerCell("\u4EE3\u7801\u884C\u6570", 1200),
                headerCell("\u7C7B\u578B\u63D0\u793A", 1200),
                headerCell("\u6587\u6863\u8986\u76D6", 1200),
                headerCell("\u590D\u6742\u5EA6", 1400),
                headerCell("\u8BC4\u5206", 1200)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("SpotZoom.py", 2200, { bold: true }),
                cell("2,723", 1200, { align: AlignmentType.CENTER }),
                cell("85%", 1200, { align: AlignmentType.CENTER }),
                cell("72%", 1200, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 1400, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("42", 1200, { bold: true, color: "C62828", align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("ML \u6A21\u5757 (30\u4E2A)", 2200, { bold: true }),
                cell("14,000+", 1200, { align: AlignmentType.CENTER }),
                cell("98%", 1200, { align: AlignmentType.CENTER }),
                cell("90%", 1200, { align: AlignmentType.CENTER }),
                cell("\u4F4E-\u4E2D", 1400, { color: "2E7D32", bold: true, align: AlignmentType.CENTER }),
                cell("88", 1200, { bold: true, color: "2E7D32", align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("Config/CLI", 2200, { bold: true }),
                cell("~200", 1200, { align: AlignmentType.CENTER }),
                cell("60%", 1200, { align: AlignmentType.CENTER }),
                cell("55%", 1200, { align: AlignmentType.CENTER }),
                cell("\u4F4E", 1400, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("55", 1200, { bold: true, color: "E65100", align: AlignmentType.CENTER })
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("Scripts (JS/BAT)", 2200, { bold: true }),
                cell("~300", 1200, { align: AlignmentType.CENTER }),
                cell("N/A", 1200, { align: AlignmentType.CENTER }),
                cell("N/A", 1200, { align: AlignmentType.CENTER }),
                cell("\u4F4E", 1400, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("70", 1200, { bold: true, color: "E65100", align: AlignmentType.CENTER })
              ]
            })
          ]
        }),

        emptyLine(),
        para("\u8BC4\u4F30\u7ED3\u8BBA\uFF1AML \u6A21\u5757\u6574\u4F53\u8D28\u91CF\u4F18\u79C0\uFF0C\u7C7B\u578B\u63D0\u793A\u548C\u6587\u6863\u8986\u76D6\u7387\u5747\u8FBE\u5230\u8F83\u9AD8\u6C34\u5E73\u3002\u4E3B\u7A0B\u5E8F SpotZoom.py \u662F\u4E3B\u8981\u7684\u8D28\u91CF\u77ED\u677F\uFF0C\u62C9\u4F4E\u4E86\u6574\u4F53\u8BC4\u5206\u3002\u5EFA\u8BAE\u91CD\u70B9\u5173\u6CE8 SpotZoom.py \u7684\u91CD\u6784\u5DE5\u4F5C\u3002"),

        // ──────────────────────────────────────────────────────────────
        // SECTION 5: 修复优先级与建议
        // ──────────────────────────────────────────────────────────────
        h1("\u4E94\u3001\u4FEE\u590D\u4F18\u5148\u7EA7\u4E0E\u5EFA\u8BAE"),
        para("\u4EE5\u4E0B\u4E3A Top 10 \u4FEE\u590D\u5EFA\u8BAE\uFF0C\u6309\u4F18\u5148\u7EA7\u6392\u5E8F\uFF1A"),
        emptyLine(),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [900, 700, 2200, 900, 900, 4000],
          rows: [
            new TableRow({
              cantSplit: true,
              children: [
                headerCell("\u4F18\u5148\u7EA7", 900),
                headerCell("ID", 700),
                headerCell("\u63CF\u8FF0", 2200),
                headerCell("\u5DE5\u4F5C\u91CF", 900),
                headerCell("\u5F71\u54CD", 900),
                headerCell("\u5EFA\u8BAE\u4FEE\u590D\u65B9\u6848", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P0", 900, { bold: true, color: sevColor("CRITICAL"), align: AlignmentType.CENTER }),
                cell("C004", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("XPSZAxis \u786C\u7F16\u7801\u5BC6\u7801", 2200),
                cell("\u5C0F", 900, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 900, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u4ECE\u73AF\u5883\u53D8\u91CF\u6216\u52A0\u5BC6\u914D\u7F6E\u8BFB\u53D6\u5BC6\u7801", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P0", 900, { bold: true, color: sevColor("CRITICAL"), align: AlignmentType.CENTER }),
                cell("C003", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("ML\u6A21\u5757\u5BFC\u5165\u9759\u9ED8\u541E\u9519", 2200),
                cell("\u5C0F", 900, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 900, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u6DFB\u52A0 logger.warning() \u8BB0\u5F55\u5BFC\u5165\u5931\u8D25\u539F\u56E0", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P0", 900, { bold: true, color: sevColor("CRITICAL"), align: AlignmentType.CENTER }),
                cell("H005", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("75\u5904\u9759\u9ED8except\u5757", 2200),
                cell("\u4E2D", 900, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 900, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u5C06 pass \u66FF\u6362\u4E3A logger.debug/warning", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P1", 900, { bold: true, color: sevColor("HIGH"), align: AlignmentType.CENTER }),
                cell("C001", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("SpotZoomController God Class", 2200),
                cell("\u5927", 900, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 900, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u62C6\u5206\u4E3A DetectionManager\u3001AlignmentEngine\u3001MLModuleRegistry", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P1", 900, { bold: true, color: sevColor("HIGH"), align: AlignmentType.CENTER }),
                cell("C002", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("SpotZoom.py \u6587\u4EF6\u8FC7\u957F (2723\u884C)", 2200),
                cell("\u5927", 900, { align: AlignmentType.CENTER }),
                cell("\u9AD8", 900, { color: "C62828", bold: true, align: AlignmentType.CENTER }),
                cell("\u62C6\u5206\u4E3A controller.py\u3001config.py\u3001cli.py \u7B49\u591A\u6A21\u5757", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P1", 900, { bold: true, color: sevColor("HIGH"), align: AlignmentType.CENTER }),
                cell("H006", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("4\u4E2A\u9AD8\u5708\u590D\u6742\u5EA6\u51FD\u6570", 2200),
                cell("\u4E2D", 900, { align: AlignmentType.CENTER }),
                cell("\u4E2D", 900, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u63D0\u53D6\u5B50\u51FD\u6570\u3001\u65E9\u8FD4\u56DE\u964D\u4F4E\u590D\u6742\u5EA6", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P1", 900, { bold: true, color: sevColor("HIGH"), align: AlignmentType.CENTER }),
                cell("H007", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("AlignmentConfig 50+ \u5B57\u6BB5", 2200),
                cell("\u4E2D", 900, { align: AlignmentType.CENTER }),
                cell("\u4E2D", 900, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u62C6\u5206\u4E3A PIDConfig\u3001SafetyConfig\u3001MLModuleConfig", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P2", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("H010", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("close() \u8D44\u6E90\u7BA1\u7406\u4E0D\u5B8C\u5584", 2200),
                cell("\u4E2D", 900, { align: AlignmentType.CENTER }),
                cell("\u4E2D", 900, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u91C7\u7528 context manager \u6A21\u5F0F\u7BA1\u7406\u8D44\u6E90", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P2", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("M012", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("92\u5904\u9664\u96F6\u98CE\u9669", 2200),
                cell("\u5C0F", 900, { align: AlignmentType.CENTER }),
                cell("\u4E2D", 900, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u6DFB\u52A0\u96F6\u503C\u68C0\u67E5\u6216\u4F7F\u7528 numpy.errstate", 4000)
              ]
            }),
            new TableRow({
              cantSplit: true,
              children: [
                cell("P2", 900, { bold: true, color: sevColor("MEDIUM"), align: AlignmentType.CENTER }),
                cell("M018", 700, { bold: true, align: AlignmentType.CENTER }),
                cell("\u7F3A\u5C11\u5355\u5143\u6D4B\u8BD5", 2200),
                cell("\u5927", 900, { align: AlignmentType.CENTER }),
                cell("\u4E2D", 900, { color: "E65100", bold: true, align: AlignmentType.CENTER }),
                cell("\u4E3A\u6838\u5FC3\u6A21\u5757\u6DFB\u52A0 pytest \u5355\u5143\u6D4B\u8BD5", 4000)
              ]
            })
          ]
        }),

        // ──────────────────────────────────────────────────────────────
        // SECTION 6: 下一步优化方案
        // ──────────────────────────────────────────────────────────────
        h1("\u516D\u3001\u4E0B\u4E00\u6B65\u4F18\u5316\u65B9\u6848"),

        // 6.1 Short-term
        h2("6.1 \u77ED\u671F (1-2\u5468)"),
        para("\u76EE\u6807\uFF1A\u4FEE\u590D\u6240\u6709 Critical \u95EE\u9898\uFF0C\u663E\u8457\u63D0\u5347\u4EE3\u7801\u5B89\u5168\u6027\u548C\u53EF\u8C03\u8BD5\u6027\u3002"),
        numberedItem("\u4FEE\u590D C004\uFF1A\u79FB\u9664 XPSZAxis \u786C\u7F16\u7801\u5BC6\u7801\uFF0C\u6539\u7528\u73AF\u5883\u53D8\u91CF\u6216\u52A0\u5BC6\u914D\u7F6E\u6587\u4EF6"),
        numberedItem("\u4FEE\u590D C003\uFF1A\u4E3A ML \u6A21\u5757\u5BFC\u5165\u6DFB\u52A0 logger.warning() \u8BB0\u5F55\u5F02\u5E38\u4FE1\u606F"),
        numberedItem("\u4FEE\u590D H005\uFF1A\u5C06 75 \u4E2A\u9759\u9ED8 except \u5757\u4E2D\u7684 pass \u66FF\u6362\u4E3A logger.debug()"),
        numberedItem("\u4FEE\u590D H009\uFF1A\u5C06 bat \u811A\u672C\u4E2D\u7684\u786C\u7F16\u7801\u8DEF\u5F84\u6539\u4E3A\u76F8\u5BF9\u8DEF\u5F84"),

        emptyLine(),

        // 6.2 Mid-term
        h2("6.2 \u4E2D\u671F (1\u4E2A\u6708)"),
        para("\u76EE\u6807\uFF1A\u91CD\u6784\u4E3B\u7A0B\u5E8F\u67B6\u6784\uFF0C\u63D0\u5347\u4EE3\u7801\u53EF\u7EF4\u62A4\u6027\u3002"),
        numberedItem("\u4FEE\u590D C001 + C002\uFF1A\u62C6\u5206 SpotZoom.py \u4E3A\u591A\u6A21\u5757\u67B6\u6784\uFF0C\u5C06 SpotZoomController \u62C6\u5206\u4E3A DetectionManager\u3001AlignmentEngine\u3001MLModuleRegistry"),
        numberedItem("\u4FEE\u590D H007\uFF1A\u62C6\u5206 AlignmentConfig \u4E3A\u5B50\u914D\u7F6E\u7EC4 (PIDConfig\u3001SafetyConfig\u3001MLModuleConfig)"),
        numberedItem("\u4FEE\u590D H006\uFF1A\u91CD\u6784\u9AD8\u590D\u6742\u5EA6\u51FD\u6570\uFF0C\u5C06\u5708\u590D\u6742\u5EA6\u964D\u81F3 15 \u4EE5\u4E0B"),
        numberedItem("\u4FEE\u590D M018\uFF1A\u4E3A\u6838\u5FC3\u6A21\u5757\u6DFB\u52A0 pytest \u5355\u5143\u6D4B\u8BD5\uFF0C\u8986\u76D6\u7387\u76EE\u6807 60%+"),

        emptyLine(),

        // 6.3 Long-term
        h2("6.3 \u957F\u671F (2-3\u4E2A\u6708)"),
        para("\u76EE\u6807\uFF1A\u6784\u5EFA\u5B8C\u5584\u7684\u5DE5\u7A0B\u5316\u4F53\u7CFB\uFF0C\u652F\u6491\u9879\u76EE\u957F\u671F\u7EF4\u62A4\u3002"),
        numberedItem("\u91CD\u6784\u4E3A\u6807\u51C6 Python \u5305\u7ED3\u6784\uFF0C\u4F7F\u7528 pyproject.toml \u7BA1\u7406\u4F9D\u8D56\u548C\u6784\u5EFA"),
        numberedItem("\u6DFB\u52A0 CI/CD \u6D41\u6C34\u7EBF\uFF0C\u81EA\u52A8\u5316\u6D4B\u8BD5\u3001\u4EE3\u7801\u8D28\u91CF\u68C0\u67E5\u548C\u90E8\u7F72"),
        numberedItem("\u6027\u80FD\u4F18\u5316\uFF1A\u901A\u8FC7 TensorRT/ONNX \u52A0\u90DF YOLO \u63A8\u7406\uFF0C\u5E76\u884C\u5316\u68C0\u6D4B-\u5206\u6790-\u63A7\u5236\u6D41\u6C34\u7EBF"),
        numberedItem("\u6784\u5EFA Web UI \u754C\u9762\uFF0C\u652F\u6301\u8FDC\u7A0B\u76D1\u63A7\u4E0E\u6570\u636E\u53EF\u89C6\u5316"),
      ]
    }
  ]
});

// ── Write to file ──
const OUTPUT = "e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v6_\u7EFC\u5408\u68C0\u6D4B\u62A5\u544A.docx";

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log("Report generated: " + OUTPUT);
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
