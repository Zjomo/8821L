const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat,
        HeadingLevel, BorderStyle, WidthType, ShadingType,
        PageNumber, PageBreak } = require(path.join("e:\\jupyter file\\2_Optics\\8821L", "node_modules", "docx"));

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 20 })] })]
  });
}

function cell(text, width, opts = {}) {
  const color = opts.color || "000000";
  const bold = opts.bold || false;
  const shading = opts.shading || undefined;
  const align = opts.align || AlignmentType.LEFT;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: shading ? { fill: shading, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, color, bold, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 19 })] })]
  });
}

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 200 }, children: [new TextRun({ text, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 32, color: "1F4E79" })] });
}

function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 160 }, children: [new TextRun({ text, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 26, color: "2E75B6" })] });
}

function bodyText(text) {
  return new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 21 })] });
}

function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 21 })] });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "1F4E79" },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1, keepNext: false, keepLines: false } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "SpotZoom v7 \u7efc\u5408\u68c0\u6d4b\u62a5\u544a", font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 18, color: "888888" })] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u7b2c ", font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 18 }), new TextRun({ text: " \u9875", font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 18 })] })] })
    },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 100 }, children: [new TextRun({ text: "SpotZoom \u5149\u6591\u95ed\u73af\u5bf9\u51c6\u7cfb\u7edf", bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 40, color: "1F4E79" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "v7.0 \u7efc\u5408\u68c0\u6d4b\u62a5\u544a", bold: true, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 32, color: "2E75B6" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "\u68c0\u6d4b\u65e5\u671f: 2026-05-12  |  \u9879\u76ee: 8821L  |  Python 3.10.11  |  Windows", font: { ascii: "Arial", eastAsia: "Microsoft YaHei" }, size: 20, color: "666666" })] }),

      heading1("\u4e00\u3001\u9879\u76ee\u6982\u89c8"),
      bodyText("SpotZoom \u662f\u4e00\u4e2a\u57fa\u4e8e YOLO \u6df1\u5ea6\u5b66\u4e60 + \u7535\u673a\u5e73\u53f0\u63a7\u5236\u7684\u5149\u5b66\u5b9e\u9a8c\u5ba4\u81ea\u52a8\u5316\u7cfb\u7edf\uff0c\u5b9e\u73b0\u5149\u6591\u81ea\u52a8\u68c0\u6d4b\u4e0e\u5bf9\u51c6\u3002\u672c\u62a5\u544a\u57fa\u4e8e v7.0 \u7248\u672c\uff0c\u5305\u542b\u65b0\u589e\u7684 5 \u4e2a\u521b\u65b0\u6a21\u5757\uff0c\u5bf9\u9879\u76ee\u8fdb\u884c\u5168\u9762\u7684\u9759\u6001\u4ee3\u7801\u5206\u6790\u3001\u67b6\u6784\u8bc4\u4f30\u548c\u4ee3\u7801\u8d28\u91cf\u68c0\u6d4b\u3002"),

      heading2("1.1 \u9879\u76ee\u7edf\u8ba1"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [3500, 5860],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u6307\u6807", 3500), headerCell("\u6570\u503c", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("\u4e3b\u7a0b\u5e8f SpotZoom.py", 3500), cell("2,724 \u884c / 19 \u4e2a\u7c7b / 128 \u4e2a\u51fd\u6570", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("ML \u7b97\u6cd5\u5305", 3500), cell("40 \u4e2a\u6a21\u5757 / 18,824 \u884c\u4ee3\u7801", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("Python \u6587\u4ef6\u603b\u6570", 3500), cell("42 \u4e2a\u6587\u4ef6\uff0c\u8bed\u6cd5\u68c0\u67e5\u5168\u90e8\u901a\u8fc7", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("\u5f53\u524d\u7248\u672c", 3500), cell("v7.0\uff08\u65b0\u589e 5 \u4e2a\u521b\u65b0\u6a21\u5757\uff09", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("\u6838\u5fc3\u4f9d\u8d56", 3500), cell("numpy, opencv-python, pyyaml", 5860)] }),
          new TableRow({ cantSplit: true, children: [cell("\u53ef\u9009\u4f9d\u8d56", 3500), cell("ultralytics, pyautogui, pywin32, pylablib", 5860)] }),
        ]
      }),

      heading1("\u4e8c\u3001v7.0 \u65b0\u589e\u521b\u65b0\u6a21\u5757"),
      bodyText("\u57fa\u4e8e\u5bf9 HCIPy\u3001python-control\u3001scikit-image\u3001Bluesky \u7b49\u524d\u6cbf\u5f00\u6e90\u9879\u76ee\u7684\u8c03\u7814\uff0c\u672c\u8f6e\u65b0\u589e 5 \u4e2a\u521b\u65b0\u6a21\u5757\uff1a"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2200, 2400, 2200, 2560],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u6a21\u5757", 2200), headerCell("\u7075\u611f\u6765\u6e90", 2400), headerCell("\u6838\u5fc3\u529f\u80fd", 2200), headerCell("\u521b\u65b0\u70b9", 2560)] }),
          new TableRow({ cantSplit: true, children: [cell("zernike_common.py", 2200), cell("HCIPy/AOtools", 2400), cell("\u7edf\u4e00 Zernike \u5de5\u5177", 2200), cell("\u6d88\u9664 3 \u5904\u91cd\u590d\u4ee3\u7801\uff0c\u652f\u6301 n<=6", 2560)] }),
          new TableRow({ cantSplit: true, children: [cell("lqr_controller.py", 2200), cell("python-control", 2400), cell("LQR \u6700\u4f18\u63a7\u5236", 2200), cell("DARE \u6c42\u89e3\uff0c\u591a\u53d8\u91cf\u8026\u5408\u63a7\u5236", 2560)] }),
          new TableRow({ cantSplit: true, children: [cell("image_quality_assessor.py", 2200), cell("scikit-image", 2400), cell("\u65e0\u53c2\u8003\u56fe\u50cf\u8d28\u91cf\u8bc4\u4f30", 2200), cell("5 \u7ef4\u5ea6\u7279\u5f81 + \u8d8b\u52bf\u5206\u6790 + \u8bca\u65ad", 2560)] }),
          new TableRow({ cantSplit: true, children: [cell("adaptive_noise_suppressor.py", 2200), cell("scikit-image/BM3D", 2400), cell("\u81ea\u9002\u5e94\u964d\u566a", 2200), cell("MAD \u566a\u58f0\u4f30\u8ba1 + 5 \u79cd\u7b56\u7565\u81ea\u52a8\u5207\u6362", 2560)] }),
          new TableRow({ cantSplit: true, children: [cell("convergence_predictor.py", 2200), cell("python-control/Bluesky", 2400), cell("\u6536\u655b\u9884\u6d4b", 2200), cell("\u6307\u6570\u8870\u51cf\u62df\u5408 + \u65e9\u671f\u7ec8\u6b62 + \u5e72\u9884\u5efa\u8bae", 2560)] }),
        ]
      }),

      heading1("\u4e09\u3001\u672c\u8f6e\u68c0\u6d4b\u7ed3\u679c"),
      heading2("3.1 \u6784\u5efa\u4e0e\u8bed\u6cd5\u68c0\u67e5"),
      bodyText("\u2714 \u5168\u90e8 42 \u4e2a Python \u6587\u4ef6\u8bed\u6cd5\u68c0\u67e5\u901a\u8fc7\uff0c\u65e0\u8bed\u6cd5\u9519\u8bef\u3002"),
      bodyText("\u2714 \u6838\u5fc3\u4f9d\u8d56 (numpy, cv2, yaml) \u5168\u90e8\u53ef\u7528\u3002"),
      bodyText("\u26a0 \u53ef\u9009\u4f9d\u8d56 (ultralytics, pyautogui, pywin32, pylablib) \u672a\u5b89\u88c5\uff0c\u5f71\u54cd YOLO \u63a8\u7406\u548c\u786c\u4ef6\u63a7\u5236\uff0c\u4f46\u4e0d\u5f71\u54cd ML \u7b97\u6cd5\u5305\u72ec\u7acb\u8fd0\u884c\u3002"),
      bodyText("\u2714 ML \u5305 40 \u4e2a\u6a21\u5757\u5168\u90e8\u53ef\u5bfc\u5165\uff0c\u65e0\u7f3a\u5931\u6587\u4ef6\u3002"),
      bodyText("\u2714 SpotZoom.py \u4e2d 20 \u4e2a\u5ef6\u8fdf\u5bfc\u5165\u7684 ML \u6a21\u5757\u5168\u90e8\u80fd\u6b63\u786e\u89e3\u6790\u3002"),

      heading2("3.2 \u4ee3\u7801\u8d28\u91cf\u95ee\u9898\u6c47\u603b"),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [800, 2400, 1500, 2200, 2460],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("#", 800), headerCell("\u95ee\u9898", 2400), headerCell("\u5f71\u54cd\u6a21\u5757", 1500), headerCell("\u8be6\u7ec6\u63cf\u8ff0", 2200), headerCell("\u4fee\u590d\u5efa\u8bae", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("1", 800, {align: AlignmentType.CENTER}), cell("SpotZoom.py \u6587\u4ef6\u8fc7\u5927 (P0)", 2400, {shading:"FDE8E8"}), cell("\u4e3b\u7a0b\u5e8f", 1500), cell("2,724 \u884c\u5355\u6587\u4ef6\uff0c19 \u4e2a\u7c7b\u6df7\u5728\u4e00\u8d77", 2200), cell("\u62c6\u5206\u4e3a detectors/stages/controller/config", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("2", 800, {align: AlignmentType.CENTER}), cell("v7.0 \u6a21\u5757\u672a\u96c6\u6210 (P0)", 2400, {shading:"FDE8E8"}), cell("SpotZoom.py", 1500), cell("5 \u4e2a\u65b0\u6a21\u5757\u672a\u5728\u4e3b\u63a7\u5236\u5668\u4e2d\u96c6\u6210", 2200), cell("\u6dfb\u52a0\u5ef6\u8fdf\u5bfc\u5165\u548c\u8c03\u7528\u903b\u8f91", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("3", 800, {align: AlignmentType.CENTER}), cell("Zernike \u4ee3\u7801\u91cd\u590d (P1)", 2400, {shading:"FFF3CD"}), cell("modal_controller, zernike_analyzer", 1500), cell("_NOLL_TO_NM \u5728 3 \u4e2a\u6587\u4ef6\u4e2d\u91cd\u590d", 2200), cell("\u91cd\u6784\u4e3a\u4f7f\u7528 zernike_common.py", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("4", 800, {align: AlignmentType.CENTER}), cell("Strehl \u516c\u5f0f\u4e0d\u4e00\u81f4 (P1)", 2400, {shading:"FFF3CD"}), cell("phase_retrieval, zernike", 1500), cell("Mar\u00e9chal \u8fd1\u4f3c\u516c\u5f0f\u4e24\u5904\u4e0d\u540c", 2200), cell("\u7edf\u4e00\u4e3a exp(-(2pi*sigma)^2)", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("5", 800, {align: AlignmentType.CENTER}), cell("AlignmentConfig \u8fc7\u5927 (P1)", 2400, {shading:"FFF3CD"}), cell("SpotZoom.py", 1500), cell("60+ \u5b57\u6bb5\u6df7\u5408\u914d\u7f6e", 2200), cell("\u6309\u529f\u80fd\u5206\u7ec4\u4e3a\u5b50\u914d\u7f6e\u7c7b", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("6", 800, {align: AlignmentType.CENTER}), cell("\u5df2\u5f03\u7528\u6587\u4ef6\u672a\u6e05\u7406 (P1)", 2400, {shading:"FFF3CD"}), cell("correct_robot.py", 1500), cell("480 \u884c\u5df2\u5f03\u7528\u4ee3\u7801", 2200), cell("\u5220\u9664\u6216\u79fb\u5165 archive/", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("7", 800, {align: AlignmentType.CENTER}), cell("\u7f3a\u5c11 reset() (P2)", 2400, {shading:"E8F5E9"}), cell("4 \u4e2a\u6a21\u5757", 1500), cell("event_bus, spot_quality, subpixel_centroid, zernike_analyzer", 2200), cell("\u8865\u5145 reset() \u65b9\u6cd5", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("8", 800, {align: AlignmentType.CENTER}), cell("\u7f3a\u5c11 logging (P2)", 2400, {shading:"E8F5E9"}), cell("6 \u4e2a\u6a21\u5757", 1500), cell("adaptive_gain, kalman_tracker \u7b49", 2200), cell("\u6dfb\u52a0 logger \u548c\u65e5\u5fd7", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("9", 800, {align: AlignmentType.CENTER}), cell("\u786c\u7f16\u7801\u56fe\u50cf\u5c3a\u5bf8 (P2)", 2400, {shading:"E8F5E9"}), cell("active_learning_collector", 1500), cell("640x480 \u786c\u7f16\u7801", 2200), cell("\u4ece\u56fe\u50cf\u5b9e\u9645\u5c3a\u5bf8\u83b7\u53d6", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("10", 800, {align: AlignmentType.CENTER}), cell("\u5360\u4f4d\u5b9e\u73b0 (P2)", 2400, {shading:"E8F5E9"}), cell("backend_accelerator", 1500), cell("_run_inference() \u4f7f\u7528 sleep \u6a21\u62df", 2200), cell("\u5b9e\u73b0\u771f\u5b9e\u63a8\u7406\u57fa\u51c6\u6d4b\u8bd5", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("11", 800, {align: AlignmentType.CENTER}), cell("\u8fdf\u6ede\u8865\u507f\u672a\u5b9e\u73b0 (P3)", 2400, {shading:"F0F0F0"}), cell("focus_search", 1500), cell("hysteresis_compensation \u672a\u4f7f\u7528", 2200), cell("\u5b9e\u73b0\u8fdf\u6ede\u8865\u507f\u903b\u8f91", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("12", 800, {align: AlignmentType.CENTER}), cell("\u5386\u53f2\u6570\u636e\u5b58\u50a8\u6548\u7387 (P3)", 2400, {shading:"F0F0F0"}), cell("\u591a\u4e2a\u6a21\u5757", 1500), cell("list+\u622a\u65ad\u800c\u975e deque", 2200), cell("\u66ff\u6362\u4e3a collections.deque", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("13", 800, {align: AlignmentType.CENTER}), cell("package.json \u4e0d\u5b8c\u6574 (P3)", 2400, {shading:"F0F0F0"}), cell("\u6839\u76ee\u5f55", 1500), cell("\u7f3a\u5c11 name, version, scripts", 2200), cell("\u8865\u5145\u6807\u51c6 npm \u5143\u6570\u636e", 2460)] }),
          new TableRow({ cantSplit: true, children: [cell("14", 800, {align: AlignmentType.CENTER}), cell("Zernike \u9636\u6570\u9650\u5236 (P3)", 2400, {shading:"F0F0F0"}), cell("3 \u4e2a\u6a21\u5757", 1500), cell("\u4ec5\u652f\u6301 n<=4\uff0czernike_common \u5df2\u6269\u5c55\u5230 n<=6", 2200), cell("\u66f4\u65b0\u5176\u4ed6\u6a21\u5757\u4f7f\u7528 zernike_common", 2460)] }),
        ]
      }),

      heading1("\u56db\u3001\u67b6\u6784\u8bc4\u4f30"),
      heading2("4.1 \u6a21\u5757\u8026\u5408\u5206\u6790"),
      bodyText("\u2714 ML \u5305\u5185\u90e8\u6a21\u5757\u95f4\u65e0\u4ea4\u53c9\u5bfc\u5165\uff0c\u8026\u5408\u5ea6\u4f4e\uff0c\u8bbe\u8ba1\u826f\u597d\u3002"),
      bodyText("\u2714 SpotZoom.py \u901a\u8fc7\u5ef6\u8fdf\u5bfc\u5165 + \u5bb9\u9519\u673a\u5236\u96c6\u6210 ML \u6a21\u5757\uff0c\u5b9e\u73b0\u4e86\u4f18\u96c5\u964d\u7ea7\u3002"),
      bodyText("\u26a0 \u4e3b\u7a0b\u5e8f\u4e0e ML \u5305\u4e4b\u95f4\u7f3a\u5c11\u6b63\u5f0f\u7684\u62bd\u8c61\u63a5\u53e3\u5c42\u3002"),

      heading2("4.2 \u6027\u80fd\u98ce\u9669"),
      bodyText("\u26a0 \u591a\u4e2a\u6a21\u5757\u4f7f\u7528 list + \u624b\u52a8\u622a\u65ad\u5b58\u50a8\u5386\u53f2\u6570\u636e\uff0c\u9ad8\u9891\u8c03\u7528\u65f6\u53ef\u80fd\u4ea7\u751f\u6027\u80fd\u95ee\u9898\u3002"),
      bodyText("\u26a0 \u5b9e\u65f6\u6027\u80fd\u76d1\u63a7\u548c\u5f02\u5e38\u68c0\u6d4b\u540c\u65f6\u8fd0\u884c\u65f6\u53ef\u80fd\u4ea7\u751f\u5f00\u9500\u3002"),
      bodyText("\u2714 \u5ef6\u8fdf\u5bfc\u5165\u673a\u5236\u907f\u514d\u4e86\u4e0d\u5fc5\u8981\u6a21\u5757\u7684\u52a0\u8f7d\u5f00\u9500\u3002"),

      heading2("4.3 \u56de\u5f52\u98ce\u9669"),
      bodyText("\u26a0 \u4e3b\u7a0b\u5e8f\u6587\u4ef6\u8fc7\u5927 (2,724 \u884c)\uff0c\u4efb\u4f55\u4fee\u6539\u90fd\u6709\u8f83\u9ad8\u56de\u5f52\u98ce\u9669\u3002"),
      bodyText("\u26a0 \u7f3a\u5c11\u5355\u5143\u6d4b\u8bd5\uff0c\u65e0\u6cd5\u81ea\u52a8\u9a8c\u8bc1\u6a21\u5757\u4fee\u6539\u4e0d\u5f71\u54cd\u5176\u4ed6\u6a21\u5757\u3002"),
      bodyText("\u2714 ML \u5305\u6a21\u5757\u95f4\u65e0\u8026\u5408\uff0c\u5355\u4e2a\u6a21\u5757\u4fee\u6539\u56de\u5f52\u98ce\u9669\u4f4e\u3002"),

      heading1("\u4e94\u3001\u4e0b\u4e00\u6b65\u4f18\u5316\u65b9\u6848"),
      heading2("5.1 \u7b2c\u4e00\u9636\u6bb5\uff1a\u7d27\u6025\u4fee\u590d (1-2 \u5929)"),
      bullet("\u5c06 v7.0 \u7684 5 \u4e2a\u65b0\u6a21\u5757\u96c6\u6210\u5230 SpotZoomController \u4e2d"),
      bullet("\u91cd\u6784 modal_controller.py \u548c zernike_analyzer.py \u4f7f\u7528 zernike_common.py"),
      bullet("\u7edf\u4e00 Strehl \u6bd4\u8ba1\u7b97\u516c\u5f0f"),
      bullet("\u4e3a 4 \u4e2a\u7f3a\u5c11 reset() \u7684\u6a21\u5757\u8865\u5145\u65b9\u6cd5"),

      heading2("5.2 \u7b2c\u4e8c\u9636\u6bb5\uff1a\u67b6\u6784\u4f18\u5316 (3-5 \u5929)"),
      bullet("\u62c6\u5206 SpotZoom.py: detectors.py, stages.py, controller.py, config.py"),
      bullet("\u5c06 AlignmentConfig \u6309\u529f\u80fd\u5206\u7ec4\u4e3a\u5b50\u914d\u7f6e\u7c7b"),
      bullet("\u6e05\u7406\u5df2\u5f03\u7528\u7684 correct_robot.py"),
      bullet("\u4e3a 6 \u4e2a\u7f3a\u5c11 logging \u7684\u6a21\u5757\u6dfb\u52a0\u65e5\u5fd7"),

      heading2("5.3 \u7b2c\u4e09\u9636\u6bb5\uff1a\u8d28\u91cf\u63d0\u5347 (5-7 \u5929)"),
      bullet("\u6dfb\u52a0 pytest \u5355\u5143\u6d4b\u8bd5"),
      bullet("\u5c06\u5386\u53f2\u6570\u636e\u5b58\u50a8\u4ece list \u66ff\u6362\u4e3a deque"),
      bullet("\u4fee\u590d active_learning_collector \u786c\u7f16\u7801\u95ee\u9898"),
      bullet("\u5b9e\u73b0 backend_accelerator \u771f\u5b9e\u63a8\u7406\u57fa\u51c6\u6d4b\u8bd5"),
      bullet("\u5b8c\u5584 package.json \u5143\u6570\u636e"),

      heading1("\u516d\u3001\u603b\u7ed3"),
      bodyText("\u672c\u8f6e\u68c0\u6d4b\u8986\u76d6\u4e86\u9879\u76ee\u7684\u6784\u5efa\u3001\u4ee3\u7801\u8d28\u91cf\u3001\u67b6\u6784\u8bbe\u8ba1\u548c\u521b\u65b0\u6a21\u5757\u96c6\u6210\u56db\u4e2a\u7ef4\u5ea6\u3002\u9879\u76ee\u6574\u4f53\u4ee3\u7801\u8d28\u91cf\u8f83\u9ad8\uff0cML \u5305\u8bbe\u8ba1\u4f18\u96c5\uff0c\u6a21\u5757\u95f4\u4f4e\u8026\u5408\u3002\u4e3b\u8981\u95ee\u9898\u96c6\u4e2d\u5728\u4e3b\u7a0b\u5e8f\u6587\u4ef6\u8fc7\u5927\u3001\u90e8\u5206\u4ee3\u7801\u91cd\u590d\u548c\u65b0\u6a21\u5757\u5c1a\u672a\u96c6\u6210\u3002\u5efa\u8bae\u6309\u4e09\u9636\u6bb5\u4f18\u5316\u65b9\u6848\u9010\u6b65\u5b9e\u65bd\u3002"),

      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2340, 2340, 2340, 2340],
        rows: [
          new TableRow({ cantSplit: true, children: [headerCell("\u4f18\u5148\u7ea7", 2340), headerCell("\u95ee\u9898\u6570", 2340), headerCell("\u5f71\u54cd\u8303\u56f4", 2340), headerCell("\u5efa\u8bae\u65f6\u95f4", 2340)] }),
          new TableRow({ cantSplit: true, children: [cell("P0 \u4e25\u91cd", 2340, {bold:true, color:"CC0000", shading:"FDE8E8"}), cell("2", 2340, {align: AlignmentType.CENTER}), cell("\u4e3b\u7a0b\u5e8f + \u96c6\u6210", 2340), cell("1-2 \u5929", 2340)] }),
          new TableRow({ cantSplit: true, children: [cell("P1 \u9ad8", 2340, {bold:true, color:"CC6600", shading:"FFF3CD"}), cell("4", 2340, {align: AlignmentType.CENTER}), cell("\u591a\u6a21\u5757", 2340), cell("2-3 \u5929", 2340)] }),
          new TableRow({ cantSplit: true, children: [cell("P2 \u4e2d", 2340, {bold:true, color:"336699", shading:"E8F5E9"}), cell("4", 2340, {align: AlignmentType.CENTER}), cell("\u5355\u4e2a\u6a21\u5757", 2340), cell("3-5 \u5929", 2340)] }),
          new TableRow({ cantSplit: true, children: [cell("P3 \u4f4e", 2340, {bold:true, color:"666666", shading:"F0F0F0"}), cell("4", 2340, {align: AlignmentType.CENTER}), cell("\u975e\u5173\u952e\u8def\u5f84", 2340), cell("\u540e\u7eed\u8fed\u4ee3", 2340)] }),
        ]
      }),
    ]
  }]
});

const outPath = path.join("e:\\jupyter file\\2_Optics\\8821L", "SpotZoom_v7_\u7efc\u5408\u68c0\u6d4b\u62a5\u544a.docx");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("Report generated: " + outPath);
});
