const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: { size: 20, ascii: "Arial", eastAsia: "Microsoft YaHei" } })] })]
  });
}

function cell(text, width, opts = {}) {
  const color = opts.color || "000000";
  const bold = opts.bold || false;
  const bg = opts.bg || undefined;
  const align = opts.align || AlignmentType.LEFT;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: bg ? { fill: bg, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, color, bold, font: { size: 18, ascii: "Arial", eastAsia: "Microsoft YaHei" } })] })]
  });
}

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 180 }, children: [new TextRun({ text, bold: true, font: { size: 30, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "1F4E79" })] });
}

function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text, bold: true, font: { size: 24, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "2E75B6" })] });
}

function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text, font: { size: 20, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: opts.color || "333333", bold: opts.bold || false })] });
}

function bulletItem(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: { size: 20, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "333333" })]
  });
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 20 } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0, keepNext: false, keepLines: false } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: false, keepLines: false } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1200, right: 1200, bottom: 1200, left: 1200 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "SpotZoom \u9879\u76EE\u68C0\u6D4B\u62A5\u544A", font: { size: 16, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "999999" })] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u7B2C ", font: { size: 16 } }), new TextRun({ children: [PageNumber.CURRENT], font: { size: 16 } }), new TextRun({ text: " \u9875", font: { size: 16 } })] })] })
    },
    children: [
      new Paragraph({ spacing: { before: 3000 }, children: [] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "SpotZoom \u9879\u76EE\u5168\u9762\u68C0\u6D4B\u62A5\u544A", bold: true, font: { size: 44, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "1F4E79" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "\u5149\u6591\u95ED\u73AF\u5BF9\u51C6\u7CFB\u7EDF (8821L)", font: { size: 28, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "666666" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "\u521B\u65B0\u6A21\u5757\u8865\u5145 + \u4EE3\u7801\u8D28\u91CF\u68C0\u6D4B + \u4F18\u5316\u65B9\u6848", font: { size: 22, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "888888" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u68C0\u6D4B\u65E5\u671F: 2026-05-10  |  \u9879\u76EE\u89C4\u6A21: 6,502 \u884C\u4EE3\u7801 / 12 \u4E2A ML \u6A21\u5757", font: { size: 20, ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "666666" })] }),
      new Paragraph({ children: [new PageBreak()] }),

      heading1("\u4E00\u3001\u68C0\u6D4B\u6982\u89C8"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [3500, 5826],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u68C0\u6D4B\u9879", 3500), headerCell("\u7ED3\u679C", 5826)] }),
          new TableRow({ cantSplit: true, children: [cell("\u8BED\u6CD5\u68C0\u67E5 (14 \u4E2A .py)", 3500), cell("\u5168\u90E8\u901A\u8FC7", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("ML \u6A21\u5757\u5BFC\u5165 (12 \u4E2A)", 3500), cell("\u5168\u90E8\u6210\u529F", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("ML \u6A21\u5757\u5B9E\u4F8B\u5316 (11 \u4E2A)", 3500), cell("\u5168\u90E8\u901A\u8FC7", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u4F9D\u8D56\u5E93\u68C0\u67E5", 3500), cell("numpy/opencv/pyyaml/pylablib/pyautogui/pywin32/ultralytics \u5168\u90E8\u5C31\u7EEA", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("SpotZoom.py \u8BED\u6CD5", 3500), cell("\u901A\u8FC7 (2,490 \u884C)", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u65B0\u589E\u521B\u65B0\u6A21\u5757", 3500), cell("5 \u4E2A\u6A21\u5757\u5DF2\u96C6\u6210\u5230\u4E3B\u7A0B\u5E8F", 5826, { color: "2E7D32", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u65E2\u6709 Bug \u4FEE\u590D", 3500), cell("adaptive_gain.py \u53C2\u6570\u540D\u9519\u8BEF\u5DF2\u4FEE\u590D", 5826, { color: "E65100", bold: true })] }),
          new TableRow({ cantSplit: true, children: [cell("\u5355\u5143\u6D4B\u8BD5", 3500), cell("\u7F3A\u5931 (\u65E0\u4EFB\u4F55\u6D4B\u8BD5\u6587\u4EF6)", 5826, { color: "C62828", bold: true })] }),
        ]
      }),

      heading1("\u4E8C\u3001\u65B0\u589E\u521B\u65B0\u6A21\u5757 (v2.0)"),
      para("\u57FA\u4E8E\u524D\u6CBF\u5F00\u6E90\u9879\u76EE\u8C03\u7814\uFF0C\u672C\u8F6E\u65B0\u589E 5 \u4E2A\u521B\u65B0\u6A21\u5757\uFF0C\u5DF2\u96C6\u6210\u5230 SpotZoom_Machine_Learning \u5305\u548C\u4E3B\u7A0B\u5E8F\u3002"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [2200, 2000, 2400, 2726],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u6A21\u5757", 2200), headerCell("\u7075\u611F\u6765\u6E90", 2000), headerCell("\u6838\u5FC3\u529F\u80FD", 2400), headerCell("\u9884\u671F\u6536\u76CA", 2726)] }),
          new TableRow({ cantSplit: true, children: [cell("ZernikeAberrationAnalyzer", 2200, { bold: true }), cell("AOtools / HCIPy", 2000), cell("Zernike \u50CF\u5DEE\u5206\u89E3", 2400), cell("\u8BCA\u65AD\u5BF9\u51C6\u504F\u5DEE\u6765\u6E90", 2726)] }),
          new TableRow({ cantSplit: true, children: [cell("ImageJacobianController", 2200, { bold: true }), cell("ViSP", 2000), cell("\u56FE\u50CF\u96C5\u53EF\u6BD4\u4F3A\u670D\u63A7\u5236", 2400), cell("\u81EA\u9002\u5E94\u6B65\u957F\u6620\u5C04", 2726)] }),
          new TableRow({ cantSplit: true, children: [cell("ModelInferenceOptimizer", 2200, { bold: true }), cell("Ultralytics / BoxMOT", 2000), cell("\u63A8\u7406\u540E\u7AEF\u4F18\u5316", 2400), cell("\u63A8\u7406\u52A0\u901F 2-5x", 2726)] }),
          new TableRow({ cantSplit: true, children: [cell("ActiveLearningCollector", 2200, { bold: true }), cell("PPAL / AL-MDN", 2000), cell("\u4E3B\u52A8\u5B66\u4E60\u6570\u636E\u91C7\u96C6", 2400), cell("\u81EA\u52A8\u6536\u96C6\u8BAD\u7EC3\u6570\u636E", 2726)] }),
          new TableRow({ cantSplit: true, children: [cell("AdaptiveFocusSearcher", 2200, { bold: true }), cell("AO \u95ED\u73AF\u63A7\u5236", 2000), cell("\u9EC4\u91D1\u5206\u5272\u7126\u641C\u7D22", 2400), cell("\u667A\u80FD\u7126\u5E73\u9762\u5B9A\u4F4D", 2726)] }),
        ]
      }),

      heading1("\u4E09\u3001\u95EE\u9898\u68C0\u6D4B\u7ED3\u679C"),
      heading2("3.1 \u9AD8\u4F18\u5148\u7EA7 (P0)"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [800, 2200, 1500, 4826],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u4F18\u5148\u7EA7", 800), headerCell("\u95EE\u9898", 2200), headerCell("\u5F71\u54CD\u6A21\u5757", 1500), headerCell("\u8BE6\u60C5\u4E0E\u4FEE\u590D\u5EFA\u8BAE", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P0", 800, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u914D\u7F6E\u9A8C\u8BC1\u7F3A\u5931", 2200, { bold: true }), cell("AlignmentConfig", 1500), cell("40+ \u5B57\u6BB5\u96F6\u9A8C\u8BC1\uFF0C\u8D1F\u5BB9\u5DEE/\u96F6\u6B65\u957F\u7B49\u975E\u6CD5\u503C\u88AB\u9759\u9ED8\u63A5\u53D7\u3002\u6DFB\u52A0 __post_init__ \u9A8C\u8BC1\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P0", 800, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u5355\u4F53\u6587\u4EF6\u8FC7\u5927", 2200, { bold: true }), cell("SpotZoom.py", 1500), cell("2,490 \u884C/21\u7C7B\u3002\u62C6\u5206\u4E3A controllers/ + hardware/ + detection/ + config.py + cli.py\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P0", 800, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u5B8C\u5168\u65E0\u6D4B\u8BD5", 2200, { bold: true }), cell("\u5168\u5C40", 1500), cell("\u4F18\u5148\u4E3A PIDController\u3001DryRunStage\u3001AlignmentConfig \u7F16\u5199\u5355\u5143\u6D4B\u8BD5\u3002", 4826)] }),
        ]
      }),
      heading2("3.2 \u4E2D\u4F18\u5148\u7EA7 (P1)"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [800, 2200, 1500, 4826],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u4F18\u5148\u7EA7", 800), headerCell("\u95EE\u9898", 2200), headerCell("\u5F71\u54CD\u6A21\u5757", 1500), headerCell("\u8BE6\u60C5\u4E0E\u4FEE\u590D\u5EFA\u8BAE", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P1", 800, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("X/Y \u8F74\u4EE3\u7801\u91CD\u590D", 2200, { bold: true }), cell("Controller", 1500), cell("\u62BD\u53D6\u4E3A _move_axis() \u65B9\u6CD5\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P1", 800, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("\u5BBD\u6CDB\u5F02\u5E38\u6355\u83B7 (30\u5904)", 2200, { bold: true }), cell("\u5168\u5C40", 1500), cell("\u7528 LOGGER.debug() \u66FF\u4EE3\u9759\u9ED8 pass\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P1", 800, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("\u9B54\u6CD5\u6570\u5B57 (~20\u5904)", 2200, { bold: true }), cell("SpotZoom.py", 1500), cell("\u63D0\u53D6\u4E3a\u547D\u540D\u5E38\u91CF\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P1", 800, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("\u6027\u80FD\u74F6\u9888", 2200, { bold: true }), cell("\u70ED\u5FAA\u73AF", 1500), cell("\u7F13\u5B58\u7070\u5EA6\u5E27\u3001\u5171\u4EAB\u5185\u5B58\u4F20\u8F93\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P1", 800, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("\u6587\u6863\u8986\u76D6 22%", 2200, { bold: true }), cell("SpotZoom.py", 1500), cell("\u8865\u5145\u6838\u5FC3\u65B9\u6CD5 docstring\uFF0C\u521B\u5EFA README.md\u3002", 4826)] }),
        ]
      }),
      heading2("3.3 \u4F4E\u4F18\u5148\u7EA7 (P2)"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [800, 2200, 1500, 4826],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u4F18\u5148\u7EA7", 800), headerCell("\u95EE\u9898", 2200), headerCell("\u5F71\u54CD\u6A21\u5757", 1500), headerCell("\u8BE6\u60C5\u4E0E\u4FEE\u590D\u5EFA\u8BAE", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P2", 800, { bold: true, color: "F57F17", align: AlignmentType.CENTER }), cell("\u7EBF\u7A0B\u5B89\u5168\u9690\u60A3", 2200, { bold: true }), cell("WorkerClient", 1500), cell("stdin/stdout \u8BFB\u5199\u9501\u7C92\u5EA6\u53EF\u6539\u8FDB\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P2", 800, { bold: true, color: "F57F17", align: AlignmentType.CENTER }), cell("\u6B7B\u4EE3\u7801", 2200, { bold: true }), cell("main()", 1500), cell("_shutdown_requested \u672A\u4F7F\u7528\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P2", 800, { bold: true, color: "F57F17", align: AlignmentType.CENTER }), cell("\u5E9F\u5F03\u4EE3\u7801\u672A\u6E05\u7406", 2200, { bold: true }), cell("correct_robot.py", 1500), cell("479 \u884C\u5DF2\u5E9F\u5F03\u4EE3\u7801\u5EFA\u8BAE\u5220\u9664\u3002", 4826)] }),
          new TableRow({ cantSplit: true, children: [cell("P2", 800, { bold: true, color: "F57F17", align: AlignmentType.CENTER }), cell("reporter \u5B88\u536B\u91CD\u590D 21\u6B21", 2200, { bold: true }), cell("Controller", 1500), cell("\u7EDF\u4E00\u4F7F\u7528 _inc_metric()\u3002", 4826)] }),
        ]
      }),

      heading1("\u56DB\u3001\u7EFC\u5408\u8BC4\u5206"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [2500, 1500, 5326],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u7EF4\u5EA6", 2500), headerCell("\u8BC4\u5206", 1500), headerCell("\u8BF4\u660E", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u67B6\u6784", 2500), cell("3/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u5355\u4F53 2,490 \u884C\uFF0C21 \u4E2A\u7C7B\u6324\u5728\u4E00\u4E2A\u6587\u4EF6", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u5F02\u5E38\u5904\u7406", 2500), cell("5/10", 1500, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("30 \u5904\u5BBD\u6CDB Exception\uFF0C\u591A\u5904\u9759\u9ED8", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u4EE3\u7801\u91CD\u590D", 2500), cell("5/10", 1500, { bold: true, color: "E65100", align: AlignmentType.CENTER }), cell("X/Y \u5BF9\u79F0\u903B\u8F91\u672A\u62BD\u8C61", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u6027\u80FD", 2500), cell("6/10", 1500, { bold: true, color: "F57F17", align: AlignmentType.CENTER }), cell("\u70ED\u5FAA\u73AF\u5E27\u62F7\u8D1D\u548C\u91CD\u590D\u8BA1\u7B97", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u7EBF\u7A0B\u5B89\u5168", 2500), cell("7/10", 1500, { bold: true, color: "2E7D32", align: AlignmentType.CENTER }), cell("\u5355\u7EBF\u7A0B\u8FD0\u884C\uFF0C\u98CE\u9669\u4F4E", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u914D\u7F6E\u9A8C\u8BC1", 2500), cell("2/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("40+ \u5B57\u6BB5\u96F6\u9A8C\u8BC1", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u5E9F\u5F03\u4EE3\u7801", 2500), cell("8/10", 1500, { bold: true, color: "2E7D32", align: AlignmentType.CENTER }), cell("\u6B63\u786E\u6807\u8BB0\uFF0C\u65E0\u6B8B\u7559\u5F15\u7528", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u9B54\u6CD5\u6570\u5B57", 2500), cell("4/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u7EA6 20 \u4E2A\u786C\u7F16\u7801\u503C", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u6D4B\u8BD5\u8986\u76D6", 2500), cell("0/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u5B8C\u5168\u6CA1\u6709\u6D4B\u8BD5", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u6587\u6863", 2500), cell("4/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER }), cell("\u4E3B\u6587\u4EF6 22%\uFF0C\u7F3A README", 5326)] }),
          new TableRow({ cantSplit: true, children: [cell("\u7EFC\u5408\u8BC4\u5206", 2500, { bold: true, bg: "E3F2FD" }), cell("4.4/10", 1500, { bold: true, color: "C62828", align: AlignmentType.CENTER, bg: "E3F2FD" }), cell("\u529F\u80FD\u5B8C\u6574\u4F46\u5DE5\u7A0B\u8D28\u91CF\u6709\u8F83\u5927\u63D0\u5347\u7A7A\u95F4", 5326, { bg: "E3F2FD" })] }),
        ]
      }),

      heading1("\u4E94\u3001\u4FEE\u590D\u5EFA\u8BAE\u4E0E\u4E0B\u4E00\u6B65\u4F18\u5316\u65B9\u6848"),
      heading2("\u7B2C\u4E00\u9636\u6BB5\uFF1A\u7D27\u6025\u4FEE\u590D (1-2 \u5929)"),
      bulletItem("\u4E3A AlignmentConfig \u6DFB\u52A0 __post_init__ \u9A8C\u8BC1\uFF0C\u62D2\u7EDD\u8D1F\u5BB9\u5DEE\u3001\u96F6\u6B65\u957F\u3001\u8D1F\u7B49\u5F85\u65F6\u95F4"),
      bulletItem("\u4FEE\u590D _import_ml_module \u4E2D except Exception \u4E3A except (ImportError, ModuleNotFoundError)"),
      bulletItem("\u4E3A Z \u8F74\u79FB\u52A8\u6DFB\u52A0 try/except \u4FDD\u62A4"),
      heading2("\u7B2C\u4E8C\u9636\u6BB5\uFF1A\u67B6\u6784\u91CD\u6784 (3-5 \u5929)"),
      bulletItem("\u62C6\u5206 SpotZoom.py \u4E3A\u591A\u4E2A\u5B50\u6A21\u5757"),
      bulletItem("\u62BD\u53D6 X/Y \u8F74\u5BF9\u9F50\u903B\u8F91\u4E3A\u901A\u7528 _move_axis()"),
      bulletItem("\u7EDF\u4E00\u4F7F\u7528 _inc_metric() \u66FF\u4EE3 21 \u5904\u91CD\u590D\u5B88\u536B"),
      bulletItem("\u63D0\u53D6\u7EA6 20 \u4E2A\u9B54\u6CD5\u6570\u5B57\u4E3A\u547D\u540D\u5E38\u91CF"),
      heading2("\u7B2C\u4E09\u9636\u6BB5\uFF1A\u6D4B\u8BD5\u4E0E\u6587\u6863 (3-5 \u5929)"),
      bulletItem("\u4E3A PIDController\u3001DryRunStage\u3001AlignmentConfig \u7F16\u5199\u5355\u5143\u6D4B\u8BD5"),
      bulletItem("\u6DFB\u52A0 DryRun \u6A21\u5F0F\u5B8C\u6574\u5BF9\u9F50\u5FAA\u73AF\u96C6\u6210\u6D4B\u8BD5"),
      bulletItem("\u8865\u5145\u6838\u5FC3\u65B9\u6CD5 docstring\uFF0C\u76EE\u6807\u8986\u76D6\u7387 60%+"),
      bulletItem("\u521B\u5EFA README.md"),
      heading2("\u7B2C\u56DB\u9636\u6BB5\uFF1A\u6027\u80FD\u4F18\u5316 (\u6301\u7EED)"),
      bulletItem("\u5C06 YOLO \u6A21\u578B\u5BFC\u51FA\u4E3A TensorRT/ONNX \u683C\u5F0F\uFF0C\u63D0\u5347\u63A8\u7406\u901F\u5EA6 2-5x"),
      bulletItem("\u4F7F\u7528\u5171\u4EAB\u5185\u5B58\u66FF\u4EE3 JPEG+base64 \u5E27\u4F20\u8F93"),
      bulletItem("\u542F\u7528 ImageJacobianController \u66FF\u4EE3\u56FA\u5B9A\u6B65\u957F"),
      bulletItem("\u542F\u7528 ActiveLearningCollector \u81EA\u52A8\u6536\u96C6\u56F0\u96BE\u6837\u672C\u4F18\u5316 YOLO"),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("SpotZoom_\u68C0\u6D4B\u62A5\u544A_20260510_v3.docx", buffer);
  console.log("Report generated successfully!");
});
