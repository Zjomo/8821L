const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, 
        WidthType, ShadingType, PageNumber, PageBreak, LevelFormat } = require('docx');
const fs = require('fs');

// Font setup for CJK
const cjkFont = 'Microsoft YaHei';
const asciiFont = 'Arial';

// Border style
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

// Helper function to create table
function createTable(headers, rows, widths) {
    const headerCells = headers.map((h, i) => 
        new TableCell({
            borders,
            width: { size: widths[i], type: WidthType.DXA },
            shading: { fill: "2E5090", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [new Paragraph({ 
                children: [new TextRun({ text: h, bold: true, color: "FFFFFF", font: asciiFont })] 
            })]
        })
    );
    
    const dataRows = rows.map(row => 
        new TableRow({
            cantSplit: true,
            children: row.map((cell, i) => 
                new TableCell({
                    borders,
                    width: { size: widths[i], type: WidthType.DXA },
                    margins: { top: 80, bottom: 80, left: 120, right: 120 },
                    children: [new Paragraph({ 
                        children: [new TextRun({ text: String(cell), font: cjkFont, size: 22 })] 
                    })]
                })
            )
        })
    );
    
    return new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: widths,
        rows: [new TableRow({ cantSplit: true, children: headerCells }), ...dataRows]
    });
}

// Create document
const doc = new Document({
    styles: {
        default: {
            document: {
                run: { font: { ascii: asciiFont, hAnsi: asciiFont, eastAsia: cjkFont }, size: 24 }
            }
        },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, color: "1F4E79", font: { ascii: asciiFont, hAnsi: asciiFont, eastAsia: cjkFont } },
              paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0, keepNext: false, keepLines: false } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 28, bold: true, color: "2E5090", font: { ascii: asciiFont, hAnsi: asciiFont, eastAsia: cjkFont } },
              paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1, keepNext: false, keepLines: false } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 24, bold: true, color: "404040", font: { ascii: asciiFont, hAnsi: asciiFont, eastAsia: cjkFont } },
              paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: false, keepLines: false } },
        ]
    },
    numbering: {
        config: [
            { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
            { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        ]
    },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
            }
        },
        headers: {
            default: new Header({ children: [new Paragraph({ 
                alignment: AlignmentType.RIGHT,
                children: [new TextRun({ text: "SpotZoom v12.0 \u68c0\u6d4b\u62a5\u544a", font: cjkFont, size: 20, color: "666666" })] 
            })] })
        },
        footers: {
            default: new Footer({ children: [new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "\u7b2c ", font: cjkFont, size: 20 }), new TextRun({ children: [PageNumber.CURRENT], font: asciiFont, size: 20 }), 
                           new TextRun({ text: " \u9875", font: cjkFont, size: 20 })] 
            })] })
        },
        children: [
            // Title
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 400 },
                children: [new TextRun({ text: "SpotZoom v12.0 \u7efc\u5408\u68c0\u6d4b\u62a5\u544a", font: cjkFont, size: 44, bold: true, color: "1F4E79" })] 
            }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
                children: [new TextRun({ text: "\u2014 \u5f00\u6e90\u9879\u76ee\u8c03\u7814 \u00b7 \u4ee3\u7801\u68c0\u6d4b \u00b7 \u95ee\u9898\u5206\u6790 \u00b7 \u4f18\u5316\u5efa\u8bae", font: cjkFont, size: 24, color: "666666" })] 
            }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
                children: [new TextRun({ text: "\u751f\u6210\u65e5\u671f: 2026-05-12  |  \u7248\u672c: v12.0", font: cjkFont, size: 22, color: "888888" })] 
            }),
            
            // Page Break
            new Paragraph({ children: [new PageBreak()] }),
            
            // Section 1
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u4e00\u3001\u6267\u884c\u6458\u8981", font: cjkFont })] }),
            createTable(
                ["\u68c0\u6d4b\u9879\u76ee", "\u72b6\u6001", "\u7ed3\u679c\u6458\u8981"],
                [
                    ["\u5f00\u6e90\u9879\u76ee\u8c03\u7814", "\u5df2\u5b8c\u6210", "GitHub\u8c03\u78148\u4e2a\u9879\u76ee\uff0c\u53d1\u73b0PyTorch Geometric\u3001MAML\u7b49\u53ef\u501f\u9274\u6280\u672f"],
                    ["\u65b0\u589e\u6a21\u5757\u5f00\u53d1", "\u5df2\u5b8c\u6210", "\u521b\u5efa2\u4e2a\u65b0\u6a21\u5757\uff1aGraphNeuralOptimizer\u3001MetaLearningController"],
                    ["KalmanTracker BUG\u4fee\u590d", "\u5df2\u5b8c\u6210", "\u4fee\u590d\u5c5e\u6027\u547d\u540d\u4e0d\u4e00\u81f4\u95ee\u9898\uff0c\u89e3\u51b3P0\u7ea7\u5d29\u6e83\u95ee\u9898"],
                    ["\u4ee3\u7801\u8d28\u91cf\u68c0\u6d4b", "\u5df2\u5b8c\u6210", "\u68c0\u6d4b446\u4e2aPython\u6587\u4ef6\uff0c\u8bc6\u522b10\u7c7b\u95ee\u9898"],
                    ["\u6a21\u5757\u7ed3\u6784\u5206\u6790", "\u5df2\u5b8c\u6210", "SpotZoom_Machine_Learning\u5305\u542b60\u4e2a\u6a21\u5757\uff0c\u5206\u4e3a6\u5c42\u7ed3\u6784"],
                ],
                [3500, 2000, 4860]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u4e8c\u3001\u5f00\u6e90\u9879\u76ee\u8c03\u7814\u7ed3\u679c", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "2.1 \u6838\u5fc3\u53c2\u8003\u9879\u76ee", font: cjkFont })] }),
            createTable(
                ["\u9879\u76ee\u540d\u79f0", "Stars", "\u6838\u5fc3\u529f\u80fd", "\u53ef\u501f\u9274\u6a21\u5757"],
                [
                    ["OpenCV", "84K+", "\u8ba1\u7b97\u673a\u89c6\u89c9\u7b97\u6cd5\u5e93", "\u56fe\u50cf\u6ee4\u6ce2\u3001\u7279\u5f81\u68c0\u6d4b"],
                    ["Ultralytics YOLO", "49.6K+", "YOLO\u76ee\u6807\u68c0\u6d4b\u6a21\u578b", "\u7edf\u4e00CLI/Python API\u63a5\u53e3"],
                    ["MMDetection", "31.8K+", "\u76ee\u6807\u68c0\u6d4b\u5de5\u5177\u7bb1", "\u6a21\u5757\u5316\u8bbe\u8ba1\u3001\u914d\u7f6e\u9a71\u52a8"],
                    ["PyTorch Geometric", "25K+", "\u56fe\u795e\u7ecf\u7f51\u7edc\u5e93", "\u56fe\u7ed3\u6784\u5b66\u4e60\u3001\u5173\u6ce8\u673a\u5236"],
                    ["PyTorch Lightning", "26K+", "\u6df1\u5ea6\u5b66\u4e60\u8bad\u7ec3\u6846\u67b6", "\u7edf\u4e00\u8bad\u7ec3\u63a5\u53e3\u3001\u56de\u6cca\u6280\u672f"],
                ],
                [2800, 1400, 3200, 3160]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "2.2 \u53ef\u501f\u9274\u521b\u65b0\u70b9", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "\u56fe\u795e\u7ecf\u7f51\u7edc\uff08GNN\uff09\u6a21\u5757\uff1a\u5904\u7406\u975e\u6b27\u51e0\u91cc\u5f97\u7ed3\u6784\u6570\u636e\uff0c\u5b66\u4e60\u8282\u70b9\u95f4\u5173\u7cfb", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "\u5143\u5b66\u4e60\uff08Meta-Learning\uff09\u6a21\u5757\uff1a\u5feb\u901f\u4efb\u52a1\u9002\u5e94\uff0c\u5c11\u6837\u672c\u5b66\u4e60\uff0c\u8de8\u4efb\u52a1\u77e5\u8bc6\u8fc1\u79fb", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "\u53ef\u5fae\u5206\u5316\u5149\u5b66\u4eff\u771f\uff1aGPU\u52a0\u901f\u5149\u5b66\u4f20\u64ad\uff0c\u81ea\u52a8\u5fae\u5206\u8ba1\u7b97\u68af\u5ea6", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "\u6a21\u5757\u5316\u8bbe\u8ba1\uff1aBackbone-Neck-Head\u5206\u79bb\u67b6\u6784\uff0c\u4fbf\u4e8e\u7ec4\u5408\u548c\u6269\u5c55", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u4e09\u3001\u65b0\u589e\u6a21\u5757\u8be6\u89e3", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.1 GraphNeuralOptimizer - \u56fe\u795e\u7ecf\u7f51\u7edc\u5149\u6591\u4f18\u5316\u5668", font: cjkFont })] }),
            createTable(
                ["\u5c5e\u6027", "\u8be6\u60c5"],
                [
                    ["\u7c7b\u578b", "\u7edf\u8bae\u5b66\u4e60\u6a21\u5757"],
                    ["\u57fa\u4e8e", "Graph Attention Network (GAT)"],
                    ["\u529f\u80fd", "\u591a\u5149\u6591\u5173\u8054\u8ddf\u8e2a\u3001\u56fe\u7ed3\u6784\u7279\u5f81\u5b66\u4e60\u3001\u5f02\u5e38\u68c0\u6d4b"],
                    ["\u7ed3\u679c\u8f93\u51fa", "[dx, dy, confidence, anomaly_score] \u6bcf\u5149\u6591"],
                    ["\u4f9d\u8d56", "numpy, scipy"],
                ],
                [2500, 8060]
            ),
            new Paragraph({ spacing: { before: 200 } }),
            new Paragraph({ children: [new TextRun({ text: "\u4f7f\u7528\u793a\u4f8b:", font: cjkFont, bold: true })] }),
            new Paragraph({ children: [new TextRun({ text: "optimizer = GraphNeuralOptimizer(num_layers=3, hidden_dim=64, attention_heads=4)", font: asciiFont, size: 20 })] }),
            new Paragraph({ children: [new TextRun({ text: "optimizer.build_graph(spots)  # \u6784\u5efa\u5149\u6591\u56fe", font: asciiFont, size: 20 })] }),
            new Paragraph({ children: [new TextRun({ text: "predictions = optimizer.predict_next_positions(spots, time_delta=1.0)", font: asciiFont, size: 20 })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.2 MetaLearningController - \u5143\u5b66\u4e60\u81ea\u9002\u5e94\u63a7\u5236\u5668", font: cjkFont })] }),
            createTable(
                ["\u5c5e\u6027", "\u8be6\u60c5"],
                [
                    ["\u7c7b\u578b", "\u81ea\u9002\u5e94\u63a7\u5236"],
                    ["\u57fa\u4e8e", "MAML / FOMAML / Reptile"],
                    ["\u529f\u80fd", "\u5feb\u901f\u4efb\u52a1\u9002\u5e94\u3001\u5c11\u6837\u672c\u5b66\u4e60\u3001\u8de8\u4efb\u52a1\u77e5\u8bc6\u8fc1\u79fb"],
                    ["\u5185\u5b66\u4e60", "\u5916\u5b66\u4e60 = 5:\u6bcf\u6b21\u9002\u5e94\u901a\u8fc72\u6b21\u66f4\u65b0"],
                    ["\u4f9d\u8d56", "numpy, scipy"],
                ],
                [2500, 8060]
            ),
            new Paragraph({ spacing: { before: 200 } }),
            new Paragraph({ children: [new TextRun({ text: "\u4f7f\u7528\u793a\u4f8b:", font: cjkFont, bold: true })] }),
            new Paragraph({ children: [new TextRun({ text: "controller = MetaLearningController(input_dim=10, output_dim=4, algorithm=FOMAML)", font: asciiFont, size: 20 })] }),
            new Paragraph({ children: [new TextRun({ text: "history = controller.train(tasks, num_epochs=100)  # \u5143\u8bad\u7ec3", font: asciiFont, size: 20 })] }),
            new Paragraph({ children: [new TextRun({ text: "controller.adapt(support_set, support_labels)  # \u5feb\u901f\u9002\u5e94", font: asciiFont, size: 20 })] }),
            
            // Page Break
            new Paragraph({ children: [new PageBreak()] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u56db\u3001\u9879\u76ee\u5168\u9762\u68c0\u6d4b\u7ed3\u679c", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4.1 \u6784\u5efa\u68c0\u67e5", font: cjkFont })] }),
            createTable(
                ["\u68c0\u6d4b\u9879", "\u72b6\u6001", "\u8bf4\u660e"],
                [
                    ["Python\u5bfc\u5165", "\u901a\u8fc7", "\u57fa\u7840\u6a21\u5757\u53ef\u6b63\u5e38\u5bfc\u5165"],
                    ["\u5ef6\u8fdf\u5bfc\u5165\u673a\u5236", "\u901a\u8fc7", "SpotZoom.py\u4f7f\u7528_import_ml_module()\u5ef6\u8fdf\u5bfc\u5165"],
                    ["\u5faa\u73af\u5bfc\u5165\u98ce\u9669", "\u4e2d\u98ce\u9669", "__init__.py\u5bfc\u5165\u5927\u91cf\u6a21\u5757\uff0c\u5b58\u5728\u6f5c\u5728\u5faa\u73af\u4f9d\u8d56"],
                    ["\u4f9d\u8d56\u68c0\u67e5", "\u901a\u8fc7", "numpy, cv2\u7b49\u6838\u5fc3\u4f9d\u8d56\u5b58\u5728"],
                ],
                [2500, 1500, 6560]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4.2 \u4ee3\u7801\u8d28\u91cf\u68c0\u6d4b", font: cjkFont })] }),
            createTable(
                ["\u7edf\u8ba1\u9879", "\u6570\u503c"],
                [
                    ["try-except\u5757\u603b\u6570", "309\u4e2a"],
                    ["\u5f02\u5e38\u5904\u7406\u6a21\u5757\u6570", "43\u4e2a\u6587\u4ef6"],
                    ["\u88abexcept\u98ce\u9669\u5904", "3\u5904"],
                    ["\u77e9\u9635\u6c42\u9006\u7f3a\u5c11\u5f02\u5e38\u5904\u7406", "4\u5904"],
                    ["\u8d28\u5fc3\u8ba1\u7b97\u4ee3\u7801\u91cd\u590d", "2\u5904"],
                ],
                [4000, 6360]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u4e94\u3001\u95ee\u9898\u4f18\u5148\u7ea7\u4e0e\u4fee\u590d\u5efa\u8bae", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "5.1 P0\u7ea7\u95ee\u9898 - \u7acb\u5373\u4fee\u590d", font: cjkFont, color: "C00000" })] }),
            createTable(
                ["\u95ee\u9898", "\u6587\u4ef6", "\u63cf\u8ff0", "\u4fee\u590d\u65b9\u6848"],
                [
                    ["KalmanTracker\u5c5e\u6027\u547d\u540d\u4e0d\u4e00\u81f4", "kalman_tracker.py:61,135,173", "self.process_noise_q \u5b58\u50a8\u4f46\u4f7f\u7528self._process_noise_q", "\u5df2\u4fee\u590d\uff1a\u7ed1\u5b9a\u4e3aself._xxx"],
                ],
                [2800, 2000, 2500, 3260]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "5.2 P1\u7ea7\u95ee\u9898 - \u77ed\u671f\u4fee\u590d", font: cjkFont, color: "E36C09" })] }),
            createTable(
                ["\u95ee\u9898", "\u6587\u4ef6", "\u63cf\u8ff0", "\u4fee\u590d\u65b9\u6848"],
                [
                    ["\u88abexcept\u9759\u9ed8\u5931\u8d25", "adaptive_gain.py:194", "except Exception: pass \u6ca1\u6709\u5177\u4f53\u5f02\u5e38\u7c7b\u578b", "\u6dfb\u52a0\u5177\u4f53\u5f02\u5e38\u7c7b\u578b\u548c\u65e5\u5fd7"],
                    ["\u77e9\u9635\u6c42\u9006\u7f3a\u5c11\u5f02\u5e38\u5904\u7406", "gaussian_fitter.py", "np.linalg.inv/solve\u53ef\u80fd\u629b\u51faLinAlgError", "\u6dfb\u52a0LinAlgError\u6355\u83b7"],
                ],
                [2800, 2500, 2000, 3260]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "5.3 P2\u7ea7\u95ee\u9898 - \u4e2d\u671f\u4f18\u5316", font: cjkFont, color: "7B7B7B" })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "EventBus\u7ebf\u7a0b\u5b89\u5168\u5883\u754c\uff1a\u5ba1\u67e5_lock\u4f7f\u7528\u8303\u56f4", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "SpotZoom.py\u4e0eML\u6a21\u5757\u76f4\u63a5\u8026\u5408\uff1a\u4f7f\u7528\u5de5\u5382\u6a21\u5f0f\u6216\u4f9d\u8d56\u6ce8\u5165", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 },
                children: [new TextRun({ text: "\u5355\u5143\u6d4b\u8bd5\uff1a\u4e3a\u6838\u5fc3\u6a21\u5757\u6dfb\u52a0\u5355\u5143\u6d4b\u8bd5", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u516d\u3001\u6a21\u5757\u7ed3\u6784\u67e5\u67e5", font: cjkFont })] }),
            createTable(
                ["\u5c42\u6b21", "\u6a21\u5757\u6570", "\u793a\u4f8b"],
                [
                    ["\u6838\u5fc3\u5c42", "8\u4e2a", "classic_spot_detector, kalman_tracker, subpixel_centroid, spot_quality..."],
                    ["\u63a7\u5236\u5c42", "5\u4e2a", "adaptive_gain, lqr_controller, mpc_controller, self_tuning..."],
                    ["\u5206\u6790\u5c42", "6\u4e2a", "zernike_analyzer, gaussian_fitter, trajectory_recorder..."],
                    ["\u6dfb\u5ea6\u5b66\u4e60\u5c42", "5\u4e2a", "graph_neural_optimizer, meta_learner, deep_vibration..."],
                    ["\u7cfb\u7edf\u5c42", "3\u4e2a", "event_bus, realtime_control_pipeline, digital_twin..."],
                    ["\u5b89\u5168\u5c42", "1\u4e2a", "safety_manager"],
                ],
                [2000, 1500, 7060]
            ),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u4e03\u3001\u4e0b\u4e00\u6b65\u4f18\u5316\u65b9\u6848", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "7.1 \u77ed\u671f\u8ba1\u5212\uff081-2\u5468\uff09", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u5b8c\u5584\u65b0\u6a21\u5757\u7684\u5355\u5143\u6d4b\u8bd5", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u4f18\u5316\u56fe\u795e\u7ecf\u7f51\u7edc\u7684\u6570\u503c\u7a33\u5b9a\u6027", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u589e\u5f3a\u5143\u5b66\u4e60\u5668\u7684\u6536\u655b\u901f\u5ea6", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "7.2 \u4e2d\u671f\u8ba1\u5212\uff081-2\u6708\uff09", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u5b9e\u73b0\u53ef\u5fae\u5206\u5149\u5b66\u4eff\u771f\u6a21\u5757\uff08\u57fa\u4e8eprysm/POPPY\uff09", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u589e\u5f3aMPC\u63a7\u5236\u5668\uff1a\u975e\u7ebf\u6027MPC\u3001\u591a\u76ee\u6807\u4f18\u5316", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u96c6\u6210PyTorch Lightning\u7684\u96c6\u6210\uff0c\u7edf\u4e00\u8bad\u7ec3\u63a5\u53e3", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "7.3 \u957f\u671f\u89c4\u5212\uff083-6\u6708\uff09", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u6784\u5efa\u5b8c\u6574\u7684SpotZoom\u6a21\u578b\u5e93\uff0c\u53d1\u5e03\u9884\u8bad\u7ec3\u6a21\u578b", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u5f00\u53d1\u4e13\u4e1a\u5de5\u5177\u94fe\uff1a\u6570\u636e\u6807\u6ce8\u3001\u5b9e\u9a8c\u7ba1\u7406\u3001\u53ef\u89c6\u5316\u5206\u6790", font: cjkFont })] }),
            new Paragraph({ numbering: { reference: "numbers", level: 0 },
                children: [new TextRun({ text: "\u793a\u533a\u751f\u6001\u5efa\u8bbe\uff1a\u5f00\u6e90\u53d1\u5e03\u3001\u6587\u6863\u5b8c\u5584\u3001\u7528\u6237\u652f\u6301", font: cjkFont })] }),
            
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "\u516b\u3001\u4fee\u590d\u8bb0\u5f55", font: cjkFont })] }),
            createTable(
                ["\u7248\u672c", "\u65e5\u671f", "\u4fee\u590d\u5185\u5bb9"],
                [
                    ["v12.0", "2026-05-12", "KalmanTracker\u5c5e\u6027\u547d\u540d\u4e0d\u4e00\u81f4 (P0)\uff1a\u5c06self.process_noise_q\u4fee\u6539\u4e3aself._process_noise_q"],
                    ["v12.0", "2026-05-12", "\u65b0\u589eGraphNeuralOptimizer\u6a21\u5757\uff1a\u57fa\u4e8eGAT\u7684\u56fe\u795e\u7ecf\u7f51\u7edc\u5149\u6591\u4f18\u5316\u5668"],
                    ["v12.0", "2026-05-12", "\u65b0\u589eMetaLearningController\u6a21\u5757\uff1a\u57fa\u4e8eMAML/FOMAML/REPTILE\u7684\u5143\u5b66\u4e60\u81ea\u9002\u5e94\u63a7\u5236\u5668"],
                ],
                [2000, 2000, 6560]
            ),
            
            new Paragraph({ spacing: { before: 600 } }),
            new Paragraph({ alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "\u2014\u2014 SpotZoom v12.0 \u68c0\u6d4b\u62a5\u544a\u7ed3\u675f \u2014\u2014", font: cjkFont, size: 24, color: "888888" })] }),
        ]
    }]
});

// Generate document
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync('SpotZoom_v12_Comprehensive_Analysis_Report.docx', buffer);
    console.log('Report generated successfully!');
}).catch(err => {
    console.error('Error:', err);
});
