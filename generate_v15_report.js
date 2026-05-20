const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat,
        HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

// 表格边框样式
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

// 创建表格单元格
function createCell(text, width, isHeader = false, colSpan = 1) {
    return new TableCell({
        borders,
        width: { size: width, type: WidthType.DXA },
        shading: isHeader ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 80, right: 80 },
        columnSpan: colSpan,
        children: [new Paragraph({
            children: [new TextRun({ text, bold: isHeader, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" }, size: 20 })]
        })]
    });
}

// 创建文档
const doc = new Document({
    styles: {
        default: {
            document: {
                run: {
                    font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" },
                    size: 21
                }
            }
        },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 32, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } },
              paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0, keepNext: false, keepLines: false } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 26, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } },
              paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1, keepNext: false, keepLines: false } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 24, bold: true, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } },
              paragraph: { spacing: { before: 120, after: 80 }, outlineLevel: 2, keepNext: false, keepLines: false } },
        ]
    },
    numbering: {
        config: [
            { reference: "bullets",
              levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
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
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({ text: "SpotZoom v15.0 综合检测报告", size: 18, color: "666666" })]
                })]
            })
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun({ text: "第 ", size: 18 }),
                        new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
                        new TextRun({ text: " 页", size: 18 })
                    ]
                })]
            })
        },
        children: [
            // 标题
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 240 },
                children: [new TextRun({ text: "SpotZoom v15.0 综合检测报告", bold: true, size: 40, font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } })]
            }),
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 400 },
                children: [new TextRun({ text: "科研前沿开源项目调研 + 代码质量分析 + 系统检测", size: 24, color: "666666", font: { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" } })]
            }),

            // 基本信息
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("一、检测基本信息")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [2340, 7020],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("检测项目", 2340, true), createCell("SpotZoom 光斑闭环对准系统", 7020)] }),
                    new TableRow({ cantSplit: true, children: [createCell("检测版本", 2340, true), createCell("v15.0 (第十五轮扩展版)", 7020)] }),
                    new TableRow({ cantSplit: true, children: [createCell("检测日期", 2340, true), createCell("2026-05-12", 7020)] }),
                    new TableRow({ cantSplit: true, children: [createCell("代码规模", 2340, true), createCell("66个模块 / 46,896行代码", 7020)] }),
                    new TableRow({ cantSplit: true, children: [createCell("Python版本", 2340, true), createCell("3.10.11", 7020)] }),
                ]
            }),
            new Paragraph({ spacing: { after: 240 }, children: [] }),

            // 第二部分：前沿开源项目调研
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("二、科研前沿开源项目调研")] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1 调研范围")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("本次调研覆盖了以下科研前沿领域的开源项目：")]
            }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("自适应光学 (Adaptive Optics): HCIPy, AOtools, SOAPY, pyRTC")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("视觉伺服 (Visual Servoing): ViSP, NSER-IBVS")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("深度学习框架: Ultralytics YOLO, PyTorch Geometric, Mamba")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("控制系统: python-control, do-mpc, ARTIQ")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("显微镜图像分析: Cellpose, StarDist, ZeroCostDL4Mic")] }),
            new Paragraph({ spacing: { after: 200 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.2 已集成创新模块")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("SpotZoom_Machine_Learning 包已集成66个创新模块，分为以下类别：")]
            }),

            // 模块分类表格
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [2340, 3510, 3510],
                rows: [
                    new TableRow({ cantSplit: true, children: [
                        createCell("模块类别", 2340, true),
                        createCell("模块数量", 3510, true),
                        createCell("代表模块", 3510, true)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("核心检测", 2340),
                        createCell("4个", 3510),
                        createCell("KalmanSpotTracker, SubPixelCentroid", 3510)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("跟踪预测", 2340),
                        createCell("5个", 3510),
                        createCell("WavefrontPredictor, MambaPredictor", 3510)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("分析评估", 2340),
                        createCell("8个", 3510),
                        createCell("ZernikeAberrationAnalyzer, XAIDiagnostic", 3510)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("控制优化", 2340),
                        createCell("9个", 3510),
                        createCell("MPCController, ImageJacobianController", 3510)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("深度学习", 2340),
                        createCell("12个", 3510),
                        createCell("MetaLearner, GraphNeuralOptimizer", 3510)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("系统仿真", 2340),
                        createCell("10个", 3510),
                        createCell("DigitalTwinSimulator, RealtimeControlPipeline", 3510)
                    ]}),
                ]
            }),
            new Paragraph({ spacing: { after: 240 }, children: [] }),

            // 第三部分：代码质量分析
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("三、代码质量分析")] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 代码规模统计")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [4680, 4680],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("指标", 4680, true), createCell("数值", 4680, true)] }),
                    new TableRow({ cantSplit: true, children: [createCell("Python模块总数", 4680), createCell("66个", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("总代码行数", 4680), createCell("46,896行", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("平均每文件行数", 4680), createCell("711行", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("最大文件", 4680), createCell("digital_twin_simulator.py (2,820行)", 4680)] }),
                ]
            }),
            new Paragraph({ spacing: { after: 200 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.2 问题汇总")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("通过静态代码分析和动态测试，发现以下问题：")]
            }),

            // P0问题
            new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("P0 - 严重问题 (需立即修复)")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [1200, 2340, 1200, 4620],
                rows: [
                    new TableRow({ cantSplit: true, children: [
                        createCell("#", 1200, true),
                        createCell("问题描述", 2340, true),
                        createCell("数量", 1200, true),
                        createCell("影响文件", 4620, true)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("1", 1200),
                        createCell("裸except捕获所有异常", 2340),
                        createCell("5处", 1200),
                        createCell("napari_adapter.py, vizarr_adapter.py, lodestar_detector.py等", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("2", 1200),
                        createCell("循环导入风险", 2340),
                        createCell("1处", 1200),
                        createCell("__init__.py 大量模块相互导入", 4620)
                    ]}),
                ]
            }),
            new Paragraph({ spacing: { after: 160 }, children: [] }),

            // P1问题
            new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("P1 - 中等问题 (建议尽快修复)")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [1200, 2340, 1200, 4620],
                rows: [
                    new TableRow({ cantSplit: true, children: [
                        createCell("#", 1200, true),
                        createCell("问题描述", 2340, true),
                        createCell("数量", 1200, true),
                        createCell("影响文件", 4620, true)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("1", 1200),
                        createCell("函数过长 (>100行)", 2340),
                        createCell("3处", 1200),
                        createCell("mpc_controller.py, xai_diagnostic.py", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("2", 1200),
                        createCell("重复代码块", 2340),
                        createCell("15处", 1200),
                        createCell("边界框裁剪逻辑多处重复", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("3", 1200),
                        createCell("未使用变量", 2340),
                        createCell("2处", 1200),
                        createCell("mpc_controller.py, graph_neural_optimizer.py", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("4", 1200),
                        createCell("硬编码魔法数字", 2340),
                        createCell("多处", 1200),
                        createCell("各模块阈值参数", 4620)
                    ]}),
                ]
            }),
            new Paragraph({ spacing: { after: 160 }, children: [] }),

            // P2问题
            new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("P2 - 轻微问题 (建议优化)")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [1200, 2340, 1200, 4620],
                rows: [
                    new TableRow({ cantSplit: true, children: [
                        createCell("#", 1200, true),
                        createCell("问题描述", 2340, true),
                        createCell("数量", 1200, true),
                        createCell("影响文件", 4620, true)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("1", 1200),
                        createCell("行过长 (>120字符)", 2340),
                        createCell("6处", 1200),
                        createCell("realtime_control_pipeline.py, safety_manager.py", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("2", 1200),
                        createCell("文档字符串不完整", 2340),
                        createCell("多处", 1200),
                        createCell("meta_learner.py等", 4620)
                    ]}),
                    new TableRow({ cantSplit: true, children: [
                        createCell("3", 1200),
                        createCell("缺少单元测试", 2340),
                        createCell("全部", 1200),
                        createCell("所有模块", 4620)
                    ]}),
                ]
            }),
            new Paragraph({ spacing: { after: 240 }, children: [] }),

            // 第四部分：系统检测结果
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("四、系统检测结果")] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1 模块导入检查")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("核心模块导入测试结果：")]
            }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [4680, 4680],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("模块", 4680, true), createCell("状态", 4680, true)] }),
                    new TableRow({ cantSplit: true, children: [createCell("SpotZoom_Machine_Learning", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("kalman_tracker", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("subpixel_centroid", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("spot_quality", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("zernike_analyzer", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("image_jacobian", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("adaptive_gain", 4680), createCell("正常", 4680)] }),
                    new TableRow({ cantSplit: true, children: [createCell("safety_manager", 4680), createCell("正常", 4680)] }),
                ]
            }),
            new Paragraph({ spacing: { after: 200 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2 依赖包检查")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [3120, 3120, 3120],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("包名", 3120, true), createCell("状态", 3120, true), createCell("版本", 3120, true)] }),
                    new TableRow({ cantSplit: true, children: [createCell("numpy", 3120), createCell("已安装", 3120), createCell("2.2.6", 3120)] }),
                    new TableRow({ cantSplit: true, children: [createCell("opencv-python", 3120), createCell("已安装", 3120), createCell("4.13.0", 3120)] }),
                    new TableRow({ cantSplit: true, children: [createCell("PyYAML", 3120), createCell("已安装", 3120), createCell("6.0.3", 3120)] }),
                    new TableRow({ cantSplit: true, children: [createCell("scikit-learn", 3120), createCell("未安装", 3120), createCell("-", 3120)] }),
                ]
            }),
            new Paragraph({ spacing: { after: 200 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3 核心功能测试")] }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [3120, 3120, 3120],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("功能模块", 3120, true), createCell("状态", 3120, true), createCell("备注", 3120, true)] }),
                    new TableRow({ cantSplit: true, children: [createCell("KalmanSpotTracker", 3120), createCell("通过", 3120), createCell("速度估计正常", 3120)] }),
                    new TableRow({ cantSplit: true, children: [createCell("SubPixelCentroid", 3120), createCell("部分", 3120), createCell("参数接口需调整", 3120)] }),
                    new TableRow({ cantSplit: true, children: [createCell("SafetyManager", 3120), createCell("部分", 3120), createCell("参数接口需调整", 3120)] }),
                ]
            }),
            new Paragraph({ spacing: { after: 240 }, children: [] }),

            // 第五部分：修复建议
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("五、修复建议与优化方案")] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 立即修复 (本周)")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("修复所有P0级别的裸except问题，改为捕获具体异常类型")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("重构__init__.py的导入结构，使用延迟导入模式避免循环依赖")] }),
            new Paragraph({ spacing: { after: 160 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.2 短期修复 (2周内)")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("拆分过长函数：mpc_controller.py的update()方法拆分")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("提取重复代码：创建utils/geometry.py统一边界框裁剪逻辑")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("删除未使用变量，清理代码")] }),
            new Paragraph({ spacing: { after: 160 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.3 中期优化 (1个月)")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("统一API接口和命名规范，所有reset()方法返回None")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("添加完整的类型注解，覆盖率目标>90%")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("提取硬编码参数为配置常量")] }),
            new Paragraph({ spacing: { after: 160 }, children: [] }),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.4 长期改进 (持续)")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("增加单元测试覆盖率，核心算法模块优先")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("完善文档字符串和注释")] }),
            new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("性能分析和优化热点代码")] }),
            new Paragraph({ spacing: { after: 240 }, children: [] }),

            // 第六部分：总结
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("六、检测总结")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("SpotZoom v15.0项目整体架构良好，已成功集成66个创新模块，覆盖机器学习、深度学习、自适应控制等前沿领域。代码规模达46,896行，模块功能丰富。")]
            }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("主要问题集中在代码质量方面：存在5处严重异常处理问题、多处重复代码和过长函数。建议按优先级逐步修复，预计2周内可完成P0和P1级别问题的修复。")]
            }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("核心功能测试显示Kalman滤波器工作正常，部分模块参数接口需要调整以保持一致性。依赖包基本满足要求，建议安装scikit-learn以支持完整功能。")]
            }),

            // 附录
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("附录：参考开源项目列表")] }),
            new Paragraph({
                spacing: { after: 120 },
                children: [new TextRun("以下为SpotZoom_Machine_Learning模块参考的科研前沿开源项目：")]
            }),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [2340, 1560, 5460],
                rows: [
                    new TableRow({ cantSplit: true, children: [createCell("项目名称", 2340, true), createCell("Stars", 1560, true), createCell("应用领域", 5460, true)] }),
                    new TableRow({ cantSplit: true, children: [createCell("Ultralytics YOLO", 2340), createCell("49.6K+", 1560), createCell("目标检测框架", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("Cellpose", 2340), createCell("1.1K+", 1560), createCell("通用细胞分割", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("StarDist", 2340), createCell("800+", 1560), createCell("星凸形对象检测", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("PyTorch Geometric", 2340), createCell("21K+", 1560), createCell("图神经网络", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("python-control", 2340), createCell("1.9K+", 1560), createCell("控制系统设计", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("do-mpc", 2340), createCell("1.2K+", 1560), createCell("模型预测控制", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("HCIPy", 2340), createCell("138", 1560), createCell("高对比度成像仿真", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("AOtools", 2340), createCell("147", 1560), createCell("自适应光学工具", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("ViSP", 2340), createCell("878", 1560), createCell("视觉伺服框架", 5460)] }),
                    new TableRow({ cantSplit: true, children: [createCell("Stable-Baselines3", 2340), createCell("10K+", 1560), createCell("强化学习算法", 5460)] }),
                ]
            }),
        ]
    }]
});

// 生成文档
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync('SpotZoom_v15_综合检测报告.docx', buffer);
    console.log('Report generated: SpotZoom_v15_综合检测报告.docx');
}).catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
