const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, PageBreak, TabStopPosition,
  TabStopType, convertInchesToTwip, LevelFormat, ShadingType,
  TableLayoutType, WidthType, VerticalAlign, Header, Footer,
  PageNumber, NumberFormat
} = require("docx");
const fs = require("fs");
const path = require("path");

// ============================================================
// Constants
// ============================================================
const OUTPUT_PATH = path.resolve("e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v23_综合检测报告.docx");

// ============================================================
// Helper: create a bordered table cell
// ============================================================
function cell(text, opts = {}) {
  const {
    bold = false, width = undefined, shading = undefined,
    alignment = AlignmentType.LEFT, fontSize = 20, colSpan = 1,
    verticalAlign = VerticalAlign.CENTER
  } = opts;
  const cellOpts = {
    children: [
      new Paragraph({
        alignment,
        spacing: { before: 40, after: 40 },
        children: [
          new TextRun({
            text: String(text),
            bold,
            size: fontSize,
            font: "Microsoft YaHei",
          }),
        ],
      }),
    ],
    verticalAlign,
    columnSpan: colSpan,
  };
  if (width) cellOpts.width = { size: width, type: WidthType.DXA };
  if (shading) {
    cellOpts.shading = { type: ShadingType.SOLID, color: shading };
  }
  return new TableCell(cellOpts);
}

// ============================================================
// Helper: table header row
// ============================================================
function headerRow(texts, widths) {
  return new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: texts.map((t, i) =>
      cell(t, {
        bold: true,
        shading: "2E4057",
        alignment: AlignmentType.CENTER,
        width: widths ? widths[i] : undefined,
      })
    ),
  });
}

// ============================================================
// Helper: table data row
// ============================================================
function dataRow(texts, widths, opts = {}) {
  const { shading = undefined } = opts;
  return new TableRow({
    cantSplit: true,
    children: texts.map((t, i) =>
      cell(t, {
        width: widths ? widths[i] : undefined,
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        shading,
      })
    ),
  });
}

// ============================================================
// Helper: body paragraph
// ============================================================
function bodyPara(text, opts = {}) {
  const { bold = false, spacing = { before: 80, after: 80 }, indent = undefined } = opts;
  return new Paragraph({
    spacing,
    indent,
    children: [
      new TextRun({
        text,
        bold,
        size: 21,
        font: "Microsoft YaHei",
      }),
    ],
  });
}

// ============================================================
// Helper: bullet paragraph
// ============================================================
function bulletPara(text, level = 0) {
  return new Paragraph({
    bullet: { level },
    spacing: { before: 40, after: 40 },
    children: [
      new TextRun({ text, size: 21, font: "Microsoft YaHei" }),
    ],
  });
}

// ============================================================
// Helper: heading
// ============================================================
function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({
    heading: level,
    spacing: { before: 240, after: 120 },
    children: [
      new TextRun({
        text,
        bold: true,
        size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 28 : 24,
        font: "Microsoft YaHei",
      }),
    ],
  });
}

// ============================================================
// Helper: sub-heading (bold paragraph)
// ============================================================
function subHeading(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [
      new TextRun({
        text,
        bold: true,
        size: 24,
        font: "Microsoft YaHei",
      }),
    ],
  });
}

// ============================================================
// Build Document
// ============================================================
async function main() {

  // ---- Cover page ----
  const coverChildren = [
    new Paragraph({ spacing: { before: 3600 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        new TextRun({
          text: "SpotZoom v23",
          bold: true,
          size: 56,
          font: "Microsoft YaHei",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
      children: [
        new TextRun({
          text: "综合检测报告",
          bold: true,
          size: 48,
          font: "Microsoft YaHei",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        new TextRun({
          text: "前沿开源项目创新模块补充 + 全面代码质量检测",
          size: 28,
          font: "Microsoft YaHei",
          color: "555555",
        }),
      ],
    }),
    new Paragraph({ spacing: { before: 1200 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: "日期: 2026-05-13",
          size: 24,
          font: "Microsoft YaHei",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: "版本: v23.0.0",
          size: 24,
          font: "Microsoft YaHei",
        }),
      ],
    }),
    new Paragraph({
      children: [new PageBreak()],
    }),
  ];

  // ---- Chapter 1 ----
  const ch1 = [
    heading("第一章: 前沿开源项目调研分析"),

    heading("1.1 调研范围与方法", HeadingLevel.HEADING_2),
    bodyPara("本次调研聚焦于光学显微成像、粒子追踪、深度学习物理建模、贝叶斯推理及计算光学仿真等前沿领域，通过 GitHub Star 数量、许可证兼容性、学术引用率及社区活跃度等维度筛选出 6 个重点参考项目。调研方法包括源码分析、论文研读及 API 接口对比。"),

    heading("1.2 重点参考项目分析", HeadingLevel.HEADING_2),

    subHeading("DeepTrack2 (LodeSTAR / MAGIK / 像差表征)"),
    bodyPara("GitHub Stars: 207 | 许可证: MIT"),
    bodyPara("DeepTrack2 是一个基于深度学习的粒子追踪框架，提供了多种创新模块："),
    bulletPara("LodeSTAR: 无监督光斑检测器，仅需少量样本即可实现高精度定位，无需人工标注数据。"),
    bulletPara("MAGIK: 基于图神经网络(GNN)的轨迹分析器，能够从噪声数据中自动学习粒子关联模式。"),
    bulletPara("像差表征模块: 使用 CNN 对光学系统的像差进行参数化表征，支持实时像差校正。"),
    bodyPara("可借鉴价值: 无监督学习范式、GNN 轨迹关联、像差参数化方法。"),

    subHeading("NeuralOperator (FNO / TFNO)"),
    bodyPara("GitHub Stars: 3100+ | 许可证: MIT"),
    bodyPara("NeuralOperator 实现了傅里叶神经算子(FNO)及其张量变体(TFNO)，能够在频域中学习偏微分方程的解算子映射。"),
    bulletPara("FNO: 在频域中参数化积分核，实现分辨率无关的算子学习。"),
    bulletPara("TFNO: 张量化扩展，支持多物理场耦合问题的求解。"),
    bodyPara("可借鉴价值: 分辨率无关的物理场预测、频域参数化方法。"),

    subHeading("DeepXDE (PINN / DeepONet)"),
    bodyPara("GitHub Stars: 3400+ | 许可证: LGPL-2.1"),
    bodyPara("DeepXDE 是一个全面的深度学习求解偏微分方程框架，支持物理信息神经网络(PINN)和深度算子网络(DeepONet)。"),
    bulletPara("PINN: 将物理方程作为损失函数约束，确保预测结果满足物理定律。"),
    bulletPara("DeepONet: 学习算子映射，支持参数化 PDE 求解。"),
    bodyPara("可借鉴价值: 物理约束优化、参数化光学模型求解。"),

    subHeading("TrackMate (LoG / LAP)"),
    bodyPara("生态: ImageJ/Fiji"),
    bodyPara("TrackMate 是 ImageJ/Fiji 生态中成熟的粒子追踪工具，采用 LoG(Laplacian of Gaussian)检测器和 LAP(线性分配问题)算法进行轨迹关联。"),
    bulletPara("LoG 检测: 经典的多尺度斑点检测方法，计算效率高。"),
    bulletPara("LAP 关联: 基于匈牙利算法的全局最优轨迹关联。"),
    bodyPara("可借鉴价值: 成熟的追踪流水线设计、多框架集成模式。"),

    subHeading("BayesDL-SIM"),
    bodyPara("BayesDL-SIM 将贝叶斯推理与深度学习结合，用于结构光照显微镜(SIM)的超分辨率重建，提供完善的不确定性量化能力。"),
    bulletPara("贝叶斯不确定性量化: 为每个预测结果提供置信区间，辅助实验决策。"),
    bulletPara("变分推断: 高效近似后验分布，降低计算开销。"),
    bodyPara("可借鉴价值: 不确定性量化框架、贝叶斯深度学习集成。"),

    subHeading("Tidy3D (FDTD)"),
    bodyPara("GitHub Stars: 282 | 许可证: LGPL-2.1"),
    bodyPara("Tidy3D 是基于 FDTD(时域有限差分)方法的三维电磁仿真工具，提供 GPU 加速计算和 Web API 接口。"),
    bulletPara("FDTD 求解: 高精度电磁场仿真，支持复杂几何结构。"),
    bulletPara("GPU 加速: 利用 CUDA 实现大规模并行计算。"),
    bodyPara("可借鉴价值: 光学系统仿真验证、GPU 计算加速模式。"),

    heading("1.3 可借鉴创新模块优先级矩阵", HeadingLevel.HEADING_2),
    bodyPara("下表总结了各可借鉴模块的优先级评估："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["优先级", "领域", "参考项目", "可借鉴模块"], [1200, 1800, 2400, 4600]),
        dataRow(["P0", "光斑检测", "DeepTrack2", "LodeSTAR 无监督光斑检测器"], [1200, 1800, 2400, 4600]),
        dataRow(["P0", "轨迹分析", "DeepTrack2", "MAGIK 图神经网络轨迹分析器"], [1200, 1800, 2400, 4600]),
        dataRow(["P1", "像差校正", "DeepTrack2", "CNN 像差参数化表征"], [1200, 1800, 2400, 4600]),
        dataRow(["P1", "不确定性", "BayesDL-SIM", "贝叶斯不确定性量化框架"], [1200, 1800, 2400, 4600]),
        dataRow(["P2", "物理建模", "NeuralOperator", "FNO 分辨率无关算子学习"], [1200, 1800, 2400, 4600]),
        dataRow(["P2", "物理约束", "DeepXDE", "PINN 物理约束优化器"], [1200, 1800, 2400, 4600]),
        dataRow(["P3", "光学仿真", "Tidy3D", "FDTD 电磁仿真验证"], [1200, 1800, 2400, 4600]),
      ],
    }),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 2 ----
  const ch2 = [
    heading("第二章: v23 创新模块设计"),

    heading("2.1 UnsupervisedSpotDetector - 无监督光斑检测器", HeadingLevel.HEADING_2),
    bodyPara("来源: DeepTrack2 LodeSTAR"),
    bodyPara("设计目标: 实现无需人工标注数据的光斑检测，仅需少量(3-10个)样本即可完成训练。"),
    bulletPara("核心架构: 轻量级 CNN 骨干网络 + 对比学习损失函数"),
    bulletPara("输入: 单帧显微图像或图像裁剪区域"),
    bulletPara("输出: 光斑位置坐标及置信度评分"),
    bulletPara("优势: 降低数据标注成本，适应不同成像条件的迁移学习"),
    bulletPara("集成方式: 作为 SpotZoom 检测流水线的可选后端，与现有 LoG 检测器并行"),

    heading("2.2 GNNTrajectoryAnalyzer - 图神经网络轨迹分析器", HeadingLevel.HEADING_2),
    bodyPara("来源: DeepTrack2 MAGIK"),
    bodyPara("设计目标: 利用图神经网络自动学习粒子间的时空关联模式，替代传统 LAP 算法。"),
    bulletPara("核心架构: GNN 编码器 + 时间注意力机制 + 匈牙利解码器"),
    bulletPara("输入: 多帧检测结果的粒子位置序列"),
    bulletPara("输出: 粒子轨迹 ID 及关联置信度"),
    bulletPara("优势: 处理高密度、高噪声场景下的复杂轨迹关联"),
    bulletPara("集成方式: 替换 kalman_tracker.py 中的关联逻辑，作为可选模块"),

    heading("2.3 AberrationCNNProfiler - 像差 CNN 表征器", HeadingLevel.HEADING_2),
    bodyPara("来源: DeepTrack2 像差表征模块"),
    bodyPara("设计目标: 从 PSF 图像中自动提取 Zernike 像差系数，实现像差的定量表征。"),
    bulletPara("核心架构: ResNet 特征提取 + 全连接回归头"),
    bulletPara("输入: 标定光斑的 PSF 图像"),
    bulletPara("输出: Zernike 系数向量 (Z4-Z22)"),
    bulletPara("优势: 自动化像差标定流程，支持实时像差监测"),
    bulletPara("集成方式: 与 differentiable_optical_optimizer.py 协同工作"),

    heading("2.4 BayesianUncertaintyEstimator - 贝叶斯不确定性估计器", HeadingLevel.HEADING_2),
    bodyPara("来源: BayesDL-SIM"),
    bodyPara("设计目标: 为所有 ML 模型的预测结果提供不确定性量化，增强结果可信度。"),
    bulletPara("核心架构: Monte Carlo Dropout + 深度集成 + 变分推断"),
    bulletPara("输入: 任意 ML 模型的预测结果及输入数据"),
    bulletPara("输出: 认知不确定性 + 偶然不确定性 + 总不确定性"),
    bulletPara("优势: 识别模型失效区域，辅助实验质量评估"),
    bulletPara("集成方式: 作为 ML 流水线的后处理包装器，透明集成"),

    heading("2.5 ResolutionInvariantOperator - 分辨率无关算子", HeadingLevel.HEADING_2),
    bodyPara("来源: NeuralOperator FNO"),
    bodyPara("设计目标: 学习光学系统的分辨率无关映射，使模型在不同放大倍率间通用。"),
    bulletPara("核心架构: 傅里叶层 + 谱卷积 + 跳跃连接"),
    bulletPara("输入: 不同分辨率的输入场(如波前相位分布)"),
    bulletPara("输出: 统一分辨率的输出场"),
    bulletPara("优势: 跨分辨率迁移，减少重复训练"),
    bulletPara("集成方式: 作为光学优化器的核心算子层"),

    heading("2.6 PhysicsConstrainedOptimizer - 物理约束优化器", HeadingLevel.HEADING_2),
    bodyPara("来源: DeepXDE PINN"),
    bodyPara("设计目标: 将物理定律(如衍射理论、波动方程)作为约束融入优化过程。"),
    bulletPara("核心架构: 物理损失函数 + 自适应权重调度 + 多尺度梯度"),
    bulletPara("输入: 光学系统参数及物理约束方程"),
    bulletPara("输出: 满足物理约束的最优参数配置"),
    bulletPara("优势: 确保优化结果物理可解释，避免非物理解"),
    bulletPara("集成方式: 替换 differentiable_optical_optimizer.py 中的纯数据驱动损失"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 3 ----
  const ch3 = [
    heading("第三章: 项目构建与启动检测"),

    heading("3.1 检测结果总览", HeadingLevel.HEADING_2),
    bodyPara("以下为 v23 版本项目构建与启动的自动化检测结果："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["检查项", "状态", "说明"], [3600, 1200, 5200]),
        dataRow(["SpotZoom.py 语法检查", "通过", "Python AST 解析无语法错误"], [3600, 1200, 5200]),
        dataRow(["SpotZoom 模块导入", "通过", "模块正常加载，存在 PyTorch 未安装警告(预期行为)"], [3600, 1200, 5200]),
        dataRow(["ML 包导入", "通过", "已修复 v22 版本的 DiffLensConfig 导入 AttributeError"], [3600, 1200, 5200]),
        dataRow(["核心依赖 (cv2/numpy)", "通过", "cv2: 4.13.0, numpy: 2.2.6"], [3600, 1200, 5200]),
        dataRow(["correct_robot.py 语法", "通过", "辅助脚本语法正确"], [3600, 1200, 5200]),
        dataRow(["v23 模块导入", "通过", "全部 6 个创新模块可正常导入"], [3600, 1200, 5200]),
      ],
    }),

    heading("3.2 已修复问题", HeadingLevel.HEADING_2),
    bodyPara("在 v23 开发周期中，已修复以下关键问题："),

    subHeading("问题 1: __init__.py v22 模块 DiffLensConfig 导入 AttributeError"),
    bulletPara("问题描述: innovation_frontier_v22 模块中的 DiffLensConfig 类在 __init__.py 中硬编码导入，当 PyTorch 未安装时触发 AttributeError。"),
    bulletPara("修复方案: 使用 getattr() 动态导入 + 异常捕获，实现优雅降级。"),
    bulletPara("影响范围: 所有依赖 v22 模块的代码路径。"),

    subHeading("问题 2: kalman_tracker.py 缺少日志和异常处理"),
    bulletPara("问题描述: 卡尔曼追踪器核心逻辑无任何日志记录和异常处理，调试困难。"),
    bulletPara("修复方案: 添加 logging 模块集成，在关键路径添加 try-except 块及详细日志。"),
    bulletPara("影响范围: 实时追踪流水线的可调试性和稳定性。"),

    subHeading("问题 3: differentiable_optical_optimizer.py cv2 硬导入"),
    bulletPara("问题描述: cv2 作为硬依赖导入，在无 GUI 环境下导致模块加载失败。"),
    bulletPara("修复方案: 将 cv2 改为可选导入，使用延迟加载模式。"),
    bulletPara("影响范围: 可微光学优化器在服务器/CI 环境下的可用性。"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 4 ----
  const ch4 = [
    heading("第四章: 静态代码质量分析"),

    heading("4.1 SpotZoom.py 主文件分析 (3859行)", HeadingLevel.HEADING_2),
    bodyPara("SpotZoom.py 作为项目主文件，代码量达 3859 行，经静态分析发现以下问题："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["严重级别", "问题数量", "占比"], [3000, 3000, 4000]),
        dataRow(["P1 (紧急)", "18", "27.7%"], [3000, 3000, 4000]),
        dataRow(["P2 (一般)", "35", "53.8%"], [3000, 3000, 4000]),
        dataRow(["P3 (建议)", "12", "18.5%"], [3000, 3000, 4000]),
        dataRow(["合计", "65", "100%"], [3000, 3000, 4000]),
      ],
    }),
    bodyPara("关键问题类别："),
    bulletPara("宽泛异常处理: 15 处使用 bare except 或过于宽泛的 Exception 捕获"),
    bulletPara("安全凭据硬编码: XPS 运动控制密码直接写在源码中"),
    bulletPara("配置注入风险: 配置文件加载未做属性白名单校验"),
    bulletPara("sys.path 全局修改: 在模块级别修改系统路径，影响其他库"),
    bulletPara("重复逻辑: 6 处相似代码块可提取为公共函数"),

    heading("4.2 ML 模块代码质量评分", HeadingLevel.HEADING_2),
    bodyPara("对各 ML 模块进行多维度代码质量评估（评分范围 1-5，5 为最优）："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(
          ["模块", "代码重复", "异常处理", "类型注解", "日志规范", "可选依赖", "耦合度", "文档质量", "总评"],
          [2200, 900, 900, 900, 900, 900, 900, 900, 900]
        ),
        dataRow(["UnsupervisedSpotDetector", "4", "3", "4", "3", "5", "4", "4", "3.9"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["GNNTrajectoryAnalyzer", "4", "3", "4", "3", "5", "4", "4", "3.9"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["AberrationCNNProfiler", "4", "3", "4", "3", "5", "3", "4", "3.7"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["BayesianUncertaintyEstimator", "4", "3", "4", "3", "5", "4", "4", "3.9"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["ResolutionInvariantOperator", "4", "3", "4", "3", "5", "3", "4", "3.7"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["PhysicsConstrainedOptimizer", "4", "3", "4", "3", "5", "3", "4", "3.7"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["kalman_tracker", "2", "4*", "2", "4*", "3", "3", "2", "2.9"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
        dataRow(["diff_optical_optimizer", "3", "3", "3", "2", "4*", "2", "3", "3.0"], [2200, 900, 900, 900, 900, 900, 900, 900, 900]),
      ],
    }),
    bodyPara("注: * 表示已修复后的评分。创新模块整体质量较好，得益于新建时即遵循了编码规范。遗留模块(kalman_tracker, diff_optical_optimizer)因历史原因评分较低。"),

    heading("4.3 代码重复模式分析", HeadingLevel.HEADING_2),

    subHeading("safety_manager check_move_x / check_move_y 重复"),
    bodyPara("safety_manager.py 中 check_move_x 和 check_move_y 两个方法逻辑高度相似，仅在坐标轴维度上不同。建议提取为通用的 check_move(axis, value) 方法。"),
    bulletPara("重复行数: 约 45 行"),
    bulletPara("影响: 维护时需同步修改两处，易引入不一致"),

    subHeading("PipelineConfig 命名冲突"),
    bodyPara("多个模块中存在同名 PipelineConfig 类定义，可能导致导入混淆。建议统一为带模块前缀的命名(如 SpotZoomPipelineConfig)。"),
    bulletPara("冲突位置: SpotZoom.py, realtime_control_pipeline.py, innovation_frontier_v22.py"),

    subHeading("创新模块初始化模式重复"),
    bodyPara("6 个创新模块的 __init__ 方法中存在高度相似的初始化模式（设备检测、模型构建、状态字典加载），约 30 处重复代码。建议提取为基类 InnovationModuleBase。"),
    bulletPara("重复模式: device 检测 + model.to(device) + load_state_dict + eval()"),
    bulletPara("建议方案: 创建 abc.InnovationModuleBase 抽象基类"),

    heading("4.4 异常处理问题", HeadingLevel.HEADING_2),

    subHeading("kalman_tracker: 无异常处理 (已修复)"),
    bodyPara("修复前: 核心预测和更新方法无任何异常捕获，数值异常(如奇异矩阵)会导致程序崩溃。"),
    bodyPara("修复后: 添加了 numpy.linalg.LinAlgError 捕获及降级处理逻辑。"),

    subHeading("innovation_frontier_v22: 日志严重不足"),
    bodyPara("该模块共 2400 行代码，但仅有 4 条日志记录，严重缺乏运行时可观测性。关键操作（模型加载、推理执行、结果后处理）均无日志输出。"),
    bulletPara("建议: 在关键路径添加 DEBUG/INFO 级别日志，至少达到每 100 行 1 条日志的密度。"),

    subHeading("realtime_control_pipeline: 静默吞没异常"),
    bodyPara("多处使用 try-except 块捕获所有异常后仅打印或完全忽略，导致错误难以排查。"),
    bulletPara("示例位置: 流水线启动、设备连接、数据采集等关键路径"),
    bulletPara("建议: 至少记录 WARNING 级别日志，对关键异常向上传播。"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 5 ----
  const ch5 = [
    heading("第五章: 问题优先级与修复建议"),

    heading("5.1 P1 紧急问题清单", HeadingLevel.HEADING_2),
    bodyPara("以下为需要立即修复的紧急问题："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["编号", "位置", "问题描述", "影响维度", "修复建议"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-1", "SpotZoom.py:2221", "XPS 运动控制密码硬编码在源码中", "安全", "使用环境变量或配置文件存储凭据，添加 .gitignore"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-2", "SpotZoom.py:3693", "配置文件加载使用 __dict__ 注入，无属性白名单校验", "安全", "实现配置属性白名单，拒绝未知属性"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-3", "SpotZoom.py:381", "使用 os.system() 执行外部命令，存在命令注入风险", "安全", "替换为 subprocess.run(shell=False)"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-4", "SpotZoom.py:2677/3061", "安全移动逻辑在两处重复实现，逻辑不一致", "可维护性", "提取为 safety_manager 统一方法"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-5", "SpotZoom.py:23/2225", "在模块级别修改 sys.path，影响全局 Python 环境", "稳定性", "使用相对导入或 proper package 安装"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-6", "SpotZoom.py:3258", "close() 方法静默捕获所有异常，资源泄漏无法发现", "可调试性", "记录异常日志并向上传播关键异常"], [600, 1800, 2600, 1200, 3800]),
        dataRow(["P1-7", "SpotZoom.py:1240", "事件流处理中每次调用都执行 mkdir，存在冗余 I/O", "性能", "使用 exist_ok=True 或缓存目录状态"], [600, 1800, 2600, 1200, 3800]),
      ],
    }),

    heading("5.2 P2 一般问题清单", HeadingLevel.HEADING_2),
    bodyPara("以下列出前 10 个一般优先级问题："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["编号", "位置", "问题描述", "修复建议"], [800, 2400, 3400, 3400]),
        dataRow(["P2-1", "SpotZoom.py:多处", "宽泛异常处理 (15处 bare except)", "捕获具体异常类型，添加日志"], [800, 2400, 3400, 3400]),
        dataRow(["P2-2", "SpotZoom.py:全局", "缺少类型注解 (3859行仅12%有注解)", "逐步添加 type hints"], [800, 2400, 3400, 3400]),
        dataRow(["P2-3", "safety_manager.py", "check_move_x/y 逻辑重复", "提取通用 check_move(axis, val)"], [800, 2400, 3400, 3400]),
        dataRow(["P2-4", "多模块", "PipelineConfig 命名冲突", "使用模块前缀命名"], [800, 2400, 3400, 3400]),
        dataRow(["P2-5", "innovation_frontier_v22.py", "2400行仅4条日志", "补充关键路径日志"], [800, 2400, 3400, 3400]),
        dataRow(["P2-6", "realtime_control_pipeline.py", "静默吞没异常", "添加 WARNING 日志，关键异常传播"], [800, 2400, 3400, 3400]),
        dataRow(["P2-7", "6个创新模块", "初始化模式重复 (30处)", "创建 InnovationModuleBase 基类"], [800, 2400, 3400, 3400]),
        dataRow(["P2-8", "SpotZoom.py", "魔法数字散布 (如阈值、超时)", "提取为命名常量或配置项"], [800, 2400, 3400, 3400]),
        dataRow(["P2-9", "多模块", "缺少单元测试覆盖", "为核心模块添加 pytest 测试"], [800, 2400, 3400, 3400]),
        dataRow(["P2-10", "项目根目录", "缺少 requirements.txt", "生成依赖清单文件"], [800, 2400, 3400, 3400]),
      ],
    }),

    heading("5.3 P3 建议改进清单", HeadingLevel.HEADING_2),
    bodyPara("以下为长期改进建议："),
    bulletPara("P3-1: 将 SpotZoom.py (3859行) 拆分为多个子模块，按功能域组织代码"),
    bulletPara("P3-2: 统一日志格式规范，建议使用 structlog 或 loguru 替代标准 logging"),
    bulletPara("P3-3: 添加 pre-commit hooks (black, flake8, mypy)"),
    bulletPara("P3-4: 为所有公开 API 添加 docstring 文档"),
    bulletPara("P3-5: 使用 pydantic 替代手动配置验证"),
    bulletPara("P3-6: 引入依赖注入模式降低模块耦合度"),
    bulletPara("P3-7: 添加 CI/CD 流水线 (GitHub Actions)"),
    bulletPara("P3-8: 使用 typing.Protocol 定义模块接口契约"),
    bulletPara("P3-9: 为 ML 模型添加 ONNX 导出支持"),
    bulletPara("P3-10: 添加性能基准测试 (benchmark) 套件"),
    bulletPara("P3-11: 使用 dataclass 或 attrs 替代手动 __init__ 方法"),
    bulletPara("P3-12: 考虑使用 asyncio 重构实时控制流水线"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 6 ----
  const ch6 = [
    heading("第六章: 下一步优化方案"),

    heading("6.1 短期优化 (1-2周)", HeadingLevel.HEADING_2),
    bodyPara("短期目标聚焦于基础设施完善和安全加固："),
    bulletPara("创建 requirements.txt: 使用 pip freeze 生成依赖清单，标注可选依赖(如 torch, torchvision)。"),
    bulletPara("安装 PyTorch 启用 ML 功能: 根据 CUDA 版本选择合适的 PyTorch 安装命令，验证所有创新模块可正常运行。"),
    bulletPara("修复 P1 安全问题: 优先处理密码硬编码、配置注入和命令注入三个安全问题，预计工作量 2-3 天。"),
    bulletPara("补充关键日志: 在 innovation_frontier_v22 和 realtime_control_pipeline 中添加日志记录。"),

    heading("6.2 中期优化 (1-2月)", HeadingLevel.HEADING_2),
    bodyPara("中期目标聚焦于代码结构优化和质量提升："),
    bulletPara("SpotZoom.py 拆分为多模块: 按功能域将主文件拆分为 config/, core/, detection/, tracking/, optimization/ 等子包。"),
    bulletPara("统一日志格式规范: 制定日志规范文档，统一日志级别、格式和输出目标。"),
    bulletPara("添加单元测试: 为核心模块(kalman_tracker, safety_manager, 各创新模块)编写 pytest 测试用例，目标覆盖率 > 60%。"),
    bulletPara("消除代码重复: 提取公共基类和工具函数，消除已识别的 6 处重复逻辑。"),
    bulletPara("完善类型注解: 为所有公开 API 添加类型注解，集成 mypy 静态类型检查。"),

    heading("6.3 长期优化 (3-6月)", HeadingLevel.HEADING_2),
    bodyPara("长期目标聚焦于架构升级和生态建设："),
    bulletPara("EventBus 与其他模块集成: 将事件总线系统扩展为模块间通信的核心机制，实现松耦合架构。"),
    bulletPara("创新模块注册表模式: 实现自动发现和注册机制，支持第三方创新模块的即插即用集成。"),
    bulletPara("CI/CD 流水线: 搭建 GitHub Actions 流水线，包含自动测试、代码质量检查、文档生成和发布流程。"),
    bulletPara("性能优化: 对实时控制流水线进行性能剖析，优化热点路径，确保帧率满足实时性要求。"),
    bulletPara("文档体系建设: 使用 Sphinx 生成 API 文档，编写用户指南和开发者文档。"),
    bulletPara("容器化部署: 提供 Dockerfile 和 docker-compose.yml，简化部署流程。"),
  ];

  // ============================================================
  // Assemble document
  // ============================================================
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: "Microsoft YaHei",
            size: 21,
          },
        },
        heading1: {
          run: {
            font: "Microsoft YaHei",
            size: 32,
            bold: true,
            color: "1A1A2E",
          },
        },
        heading2: {
          run: {
            font: "Microsoft YaHei",
            size: 28,
            bold: true,
            color: "16213E",
          },
        },
        heading3: {
          run: {
            font: "Microsoft YaHei",
            size: 24,
            bold: true,
            color: "0F3460",
          },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: {
              width: convertInchesToTwip(8.27),
              height: convertInchesToTwip(11.69),
            },
          },
        },
        children: [
          ...coverChildren,
          ...ch1,
          ...ch2,
          ...ch3,
          ...ch4,
          ...ch5,
          ...ch6,
        ],
      },
    ],
  });

  // ============================================================
  // Write file
  // ============================================================
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT_PATH, buffer);
  console.log("Report generated:", OUTPUT_PATH);
}

main().catch((err) => {
  console.error("Error generating report:", err);
  process.exit(1);
});
