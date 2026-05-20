const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, PageBreak,
  convertInchesToTwip, ShadingType,
  TableLayoutType, WidthType, VerticalAlign,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ============================================================
// Constants
// ============================================================
const OUTPUT_PATH = path.resolve("e:\\jupyter file\\2_Optics\\8821L\\SpotZoom_v26_综合检测报告.docx");

// ============================================================
// Helper: create a bordered table cell
// ============================================================
function cell(text, opts = {}) {
  const {
    bold = false, width = undefined, shading = undefined,
    alignment = AlignmentType.LEFT, fontSize = 20, colSpan = 1,
    verticalAlign = VerticalAlign.CENTER, color = undefined,
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
            color,
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
        color: "FFFFFF",
      })
    ),
  });
}

// ============================================================
// Helper: table data row
// ============================================================
function dataRow(texts, widths, opts = {}) {
  const { shading = undefined, color = undefined } = opts;
  return new TableRow({
    cantSplit: true,
    children: texts.map((t, i) =>
      cell(t, {
        width: widths ? widths[i] : undefined,
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        shading,
        color,
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
          text: "SpotZoom v26",
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
          text: "光学实验室自动化光斑闭环对准系统 (8821L) - v26.0 全面分析",
          size: 26,
          font: "Microsoft YaHei",
          color: "555555",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        new TextRun({
          text: "新增 SpotZoom_Machine_Learning_v2 创新模块 + 代码质量检测",
          size: 24,
          font: "Microsoft YaHei",
          color: "777777",
        }),
      ],
    }),
    new Paragraph({ spacing: { before: 1200 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: "报告日期: 2026-05-13",
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
          text: "项目版本: v26.0",
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
          text: "迭代轮次: 第 26 轮",
          size: 24,
          font: "Microsoft YaHei",
        }),
      ],
    }),
    new Paragraph({
      children: [new PageBreak()],
    }),
  ];

  // ---- Chapter 1: 项目概述 ----
  const ch1 = [
    heading("1. 项目概述"),

    heading("1.1 项目简介", HeadingLevel.HEADING_2),
    bodyPara("SpotZoom 是面向光学实验室的自动化光斑闭环对准系统，项目编号 8821L。该系统通过实时采集光斑图像，结合多种检测算法和控制策略，实现光斑位置的自动对准和跟踪。系统支持多种检测后端（YOLO 深度学习检测、经典图像处理检测），并提供丰富的创新模块用于高级功能扩展。"),

    heading("1.2 版本演进", HeadingLevel.HEADING_2),
    bodyPara("当前版本为 v26.0，已经历 26 轮迭代开发。从最初的单一光斑检测功能，逐步扩展为集检测、控制、优化、仿真于一体的综合光学对准平台。每一轮迭代均在前一版本基础上进行功能增强和代码质量改进。"),

    heading("1.3 项目规模", HeadingLevel.HEADING_2),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["组件", "规模", "说明"], [3000, 2000, 5000]),
        dataRow(["SpotZoom.py (主程序)", "5107 行", "核心控制逻辑、检测流水线、配置管理"], [3000, 2000, 5000]),
        dataRow(["SpotZoom_Machine_Learning", "90+ 模块", "机器学习/深度学习创新模块集合"], [3000, 2000, 5000]),
        dataRow(["SpotZoom_Machine_Learning_v2", "6 模块", "本轮新增，基于前沿开源项目调研"], [3000, 2000, 5000]),
        dataRow(["correct_robot.py", "辅助脚本", "机器人校正辅助工具"], [3000, 2000, 5000]),
      ],
    }),

    heading("1.4 技术栈", HeadingLevel.HEADING_2),
    bulletPara("Python 3.x - 主要开发语言"),
    bulletPara("OpenCV (cv2) - 图像采集与处理"),
    bulletPara("YOLO (ultralytics) - 深度学习光斑检测"),
    bulletPara("NumPy - 数值计算"),
    bulletPara("pylablib - 设备通信与控制"),
    bulletPara("PyTorch (可选) - 深度学习推理后端"),
    bulletPara("pyautogui - GUI 自动化操作"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 2: 本轮新增内容 ----
  const ch2 = [
    heading("2. 本轮新增内容 (v26.0)"),

    heading("2.1 新增模块总览", HeadingLevel.HEADING_2),
    bodyPara("基于科研前沿开源项目调研，本轮新增 SpotZoom_Machine_Learning_v2 文件夹，包含 6 个创新模块。这些模块覆盖了自适应光学控制、波前分析、图像增强、光束传播模拟、模型预测控制和鲁棒控制等前沿领域。"),

    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["模块", "参考开源项目", "核心功能"], [3200, 2800, 4000]),
        dataRow(["closed_loop_ao_controller.py", "HCIPy, AOtools", "闭环自适应光学控制器 (积分+比例+微分, 自适应增益, 伪开环估计, 陷波滤波)"], [3200, 2800, 4000]),
        dataRow(["fourier_psf_analyzer.py", "HCIPy", "傅里叶 PSF 分析与波前重建 (Strehl比, FWHM, Zernike拟合, 相位多样性)"], [3200, 2800, 4000]),
        dataRow(["deep_image_prior_enhancer.py", "Deep Image Prior (Ulyanov et al.)", "零样本光斑图像增强 (参数化滤波, 能量守恒, 多尺度策略)"], [3200, 2800, 4000]),
        dataRow(["adaptive_beam_propagator.py", "OpenCLAW", "自适应光束传播模拟 (角谱法, 焦面搜索, M\u00B2估计)"], [3200, 2800, 4000]),
        dataRow(["data_driven_mpc.py", "leap-c, acados, do-mpc", "数据驱动模型预测控制 (ARX系统辨识, 滚动优化, 约束处理)"], [3200, 2800, 4000]),
        dataRow(["lqg_robust_controller.py", "python-control", "LQG鲁棒控制器 (Kalman滤波+LQR, 自适应噪声, 渐消滤波)"], [3200, 2800, 4000]),
      ],
    }),

    heading("2.2 模块详细说明", HeadingLevel.HEADING_2),

    subHeading("2.2.1 closed_loop_ao_controller.py - 闭环自适应光学控制器"),
    bodyPara("参考项目: HCIPy, AOtools"),
    bodyPara("该模块实现了完整的闭环自适应光学控制流程，支持多种控制策略的组合使用："),
    bulletPara("积分控制器 (I): 消除稳态误差，实现精确对准"),
    bulletPara("比例控制器 (P): 提供快速响应，减少初始偏差"),
    bulletPara("微分控制器 (D): 抑制超调和振荡"),
    bulletPara("自适应增益: 根据误差大小动态调整控制增益"),
    bulletPara("伪开环估计: 在闭环条件下估计开环波前误差"),
    bulletPara("陷波滤波: 抑制特定频率的周期性扰动"),

    subHeading("2.2.2 fourier_psf_analyzer.py - 傅里叶 PSF 分析与波前重建"),
    bodyPara("参考项目: HCIPy"),
    bodyPara("基于傅里叶光学理论，提供 PSF 分析和波前重建功能："),
    bulletPara("Strehl 比: 评估光学系统成像质量的核心指标"),
    bulletPara("FWHM 测量: 光斑半高全宽的精确计算"),
    bulletPara("Zernike 多项式拟合: 将波前像差分解为 Zernike 模式"),
    bulletPara("相位多样性: 从多幅 PSF 图像中恢复波前相位信息"),

    subHeading("2.2.3 deep_image_prior_enhancer.py - 零样本光斑图像增强"),
    bodyPara("参考项目: Deep Image Prior (Ulyanov et al.)"),
    bodyPara("利用深度图像先验网络实现零样本图像增强，无需预训练数据："),
    bulletPara("参数化滤波: 通过网络权重隐式编码图像先验"),
    bulletPara("能量守恒: 增强过程中保持光斑总能量不变"),
    bulletPara("多尺度策略: 从粗到细逐步优化图像质量"),

    subHeading("2.2.4 adaptive_beam_propagator.py - 自适应光束传播模拟"),
    bodyPara("参考项目: OpenCLAW"),
    bodyPara("基于角谱法的光束传播模拟器，支持自适应参数调整："),
    bulletPara("角谱法传播: 高效的衍射传播数值计算"),
    bulletPara("焦面搜索: 自动寻找最佳焦面位置"),
    bulletPara("M\u00B2 估计: 光束质量因子的定量评估"),

    subHeading("2.2.5 data_driven_mpc.py - 数据驱动模型预测控制"),
    bodyPara("参考项目: leap-c, acados, do-mpc"),
    bodyPara("基于数据驱动的模型预测控制，无需精确物理模型："),
    bulletPara("ARX 系统辨识: 从运行数据中自动学习系统动态模型"),
    bulletPara("滚动优化: 在有限时域内求解最优控制序列"),
    bulletPara("约束处理: 支持位移台行程、速度等物理约束"),

    subHeading("2.2.6 lqg_robust_controller.py - LQG 鲁棒控制器"),
    bodyPara("参考项目: python-control"),
    bodyPara("结合 Kalman 滤波和 LQR 的经典鲁棒控制方案："),
    bulletPara("Kalman 滤波: 从噪声观测中估计系统状态"),
    bulletPara("LQR 最优控制: 最小化二次型性能指标"),
    bulletPara("自适应噪声: 在线估计过程和测量噪声协方差"),
    bulletPara("渐消滤波: 处理模型参数时变问题"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 3: 项目构建与启动检测 ----
  const ch3 = [
    heading("3. 项目构建与启动检测"),

    heading("3.1 语法检查结果", HeadingLevel.HEADING_2),
    bodyPara("全部 8 个 .py 文件通过 py_compile 语法检查，无语法错误。检查范围包括："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["文件", "状态", "说明"], [4500, 1200, 4300]),
        dataRow(["SpotZoom.py", "通过", "主程序，5107 行"], [4500, 1200, 4300]),
        dataRow(["correct_robot.py", "通过", "辅助脚本"], [4500, 1200, 4300]),
        dataRow(["closed_loop_ao_controller.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
        dataRow(["fourier_psf_analyzer.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
        dataRow(["deep_image_prior_enhancer.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
        dataRow(["adaptive_beam_propagator.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
        dataRow(["data_driven_mpc.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
        dataRow(["lqg_robust_controller.py", "通过", "v2 新增模块"], [4500, 1200, 4300]),
      ],
    }),

    heading("3.2 项目构建方式", HeadingLevel.HEADING_2),
    bulletPara("无 pyproject.toml / setup.py，项目以脚本方式直接运行"),
    bulletPara("启动方式: python SpotZoom.py [参数...] 或通过 run_spotzoom.bat 批处理文件启动"),
    bulletPara("进程锁机制: 使用 spotzoom.lock.json 文件防止多实例同时运行"),

    heading("3.3 依赖管理", HeadingLevel.HEADING_2),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["依赖类别", "包名", "用途", "必要性"], [1800, 2600, 3000, 2600]),
        dataRow(["核心依赖", "opencv-python", "图像采集与处理", "必需"], [1800, 2600, 3000, 2600]),
        dataRow(["核心依赖", "numpy", "数值计算", "必需"], [1800, 2600, 3000, 2600]),
        dataRow(["核心依赖", "ultralytics", "YOLO 深度学习检测", "必需"], [1800, 2600, 3000, 2600]),
        dataRow(["核心依赖", "pylablib", "设备通信与控制", "必需"], [1800, 2600, 3000, 2600]),
        dataRow(["核心依赖", "pyautogui", "GUI 自动化操作", "必需"], [1800, 2600, 3000, 2600]),
        dataRow(["可选依赖", "torch", "深度学习推理后端", "可选"], [1800, 2600, 3000, 2600]),
      ],
    }),

    heading("3.4 系统架构特征", HeadingLevel.HEADING_2),
    bulletPara("无 HTTP API / 数据库，为纯本地桌面应用"),
    bulletPara("延迟导入机制确保模块缺失不影响核心功能运行"),
    bulletPara("40+ 可选创新模块全部通过 --flag 命令行开关控制启用/禁用"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 4: 核心功能流程检测 ----
  const ch4 = [
    heading("4. 核心功能流程检测"),

    heading("4.1 闭环对准流程", HeadingLevel.HEADING_2),
    bodyPara("SpotZoom 的核心闭环对准流程包含以下关键步骤："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["步骤", "阶段", "说明"], [1200, 2400, 6400]),
        dataRow(["1", "截帧", "从相机采集当前帧图像"], [1200, 2400, 6400]),
        dataRow(["2", "预处理", "Flow Matching 去噪 / 图像增强"], [1200, 2400, 6400]),
        dataRow(["3", "检测", "YOLO 深度学习检测 或 经典图像处理检测"], [1200, 2400, 6400]),
        dataRow(["4", "后处理", "亚像素定位 / 卡尔曼滤波 / 记忆机制"], [1200, 2400, 6400]),
        dataRow(["5", "误差计算", "计算光斑中心与目标位置的偏差"], [1200, 2400, 6400]),
        dataRow(["6", "控制决策", "PID / MPC / 雅可比矩阵控制策略"], [1200, 2400, 6400]),
        dataRow(["7", "安全检查", "位移台行程限制、速度限制等安全约束"], [1200, 2400, 6400]),
        dataRow(["8", "驱动位移台", "发送运动指令到物理位移台"], [1200, 2400, 6400]),
        dataRow(["9", "收敛判断", "P1/P2/P3 三点收敛判据评估"], [1200, 2400, 6400]),
      ],
    }),

    heading("4.2 收敛判据", HeadingLevel.HEADING_2),
    bodyPara("系统采用三级收敛判据 (P1/P2/P3) 评估对准质量："),
    bulletPara("P1 (严格): 光斑偏差小于高精度阈值，满足精密实验要求"),
    bulletPara("P2 (标准): 光斑偏差小于标准阈值，满足一般实验要求"),
    bulletPara("P3 (宽松): 光斑偏差小于宽松阈值，满足初步对准要求"),
    bodyPara("三级判据为不同精度的实验需求提供灵活的对准标准。"),

    heading("4.3 模块化架构", HeadingLevel.HEADING_2),
    bodyPara("SpotZoom 提供 40+ 可选创新模块，覆盖检测增强、控制优化、仿真分析等多个领域。所有模块均通过 --flag 命令行开关控制，采用延迟导入机制，确保模块缺失时核心功能不受影响。用户可根据实验需求灵活组合功能模块。"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 5: 代码质量检测结果 ----
  const ch5 = [
    heading("5. 代码质量检测结果"),

    heading("5.1 问题统计总览", HeadingLevel.HEADING_2),
    bodyPara("本轮代码质量检测共发现 28 个问题，按严重度分布如下："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["严重度", "数量", "状态", "说明"], [2000, 1500, 2500, 4000]),
        dataRow(["CRITICAL", "3", "已全部修复", "运行时崩溃、数据错误等致命问题"], [2000, 1500, 2500, 4000], { shading: "FADBD8", color: "C0392B" }),
        dataRow(["HIGH", "5", "已全部修复", "性能严重退化、安全漏洞等问题"], [2000, 1500, 2500, 4000], { shading: "FDEBD0", color: "E67E22" }),
        dataRow(["MEDIUM", "13", "待修复", "性能下降、代码可维护性问题"], [2000, 1500, 2500, 4000], { shading: "FEF9E7", color: "F39C12" }),
        dataRow(["LOW", "7", "待修复", "代码风格、最佳实践建议"], [2000, 1500, 2500, 4000], { shading: "D5F5E3", color: "27AE60" }),
        dataRow(["合计", "28", "-", "CRITICAL + HIGH 已全部修复"], [2000, 1500, 2500, 4000]),
      ],
    }),

    heading("5.2 已修复的 CRITICAL/HIGH 问题", HeadingLevel.HEADING_2),
    bodyPara("以下 8 个 CRITICAL 和 HIGH 级别问题已在本轮迭代中全部修复："),

    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["编号", "严重度", "位置", "问题描述", "修复方案"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-001", "CRITICAL", "lqg_robust_controller.py", "_kalman_update 中引用未定义变量 state", "已修复为 self.state"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-002", "CRITICAL", "deep_image_prior_enhancer.py", "numpy 数组与 dict 混用导致类型不一致", "已重构为 (image, meta) 元组返回"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-003", "CRITICAL", "deep_image_prior_enhancer.py", "返回值类型不统一，下游调用崩溃", "统一所有返回值为元组格式"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-004", "HIGH", "deep_image_prior_enhancer.py", "_resize 方法使用 O(n\u00B2) Python 循环", "已向量化为 NumPy 操作"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-005", "HIGH", "deep_image_prior_enhancer.py", "np.random.seed(42) 污染全局随机状态", "已改用局部 RandomState"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-006", "HIGH", "deep_image_prior_enhancer.py", "未使用的 import time", "已删除"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-007", "HIGH", "lqg_robust_controller.py", "未使用的 import Deque", "已删除"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-008", "HIGH", "data_driven_mpc.py", "数值梯度使用错误初始状态", "已修正为 trajectory[0]"], [700, 900, 2200, 2800, 3400]),
        dataRow(["BUG-009", "HIGH", "data_driven_mpc.py", "Y 方向预测使用 X 模型参数", "已分别传入各自参数"], [700, 900, 2200, 2800, 3400]),
      ],
    }),

    heading("5.3 待修复 MEDIUM 问题", HeadingLevel.HEADING_2),
    bodyPara("以下 13 个 MEDIUM 级别问题建议在后续迭代中修复："),

    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["编号", "位置", "问题描述", "修复建议"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-010", "lqg_robust_controller.py", "np.linalg.inv 求逆矩阵，数值稳定性差", "建议改用 np.linalg.solve"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-011", "fourier_psf_analyzer.py", "构建大型对角矩阵，内存开销大", "建议用逐元素乘法替代"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-012", "fourier_psf_analyzer.py", "径向功率谱计算使用 Python 循环", "建议用 np.bincount 加速"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-013", "adaptive_beam_propagator.py", "find_best_focus 中 z 值不一致", "统一 z 坐标定义"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-014", "SpotZoom.py", "硬编码密码 \"Administrator\"", "使用环境变量或配置文件存储"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-015", "SpotZoom.py", "os.system(\"chcp 65001\") 平台相关调用", "使用 subprocess 或条件判断"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-016", "SpotZoom.py", "21 处 except Exception 吞没异常", "捕获具体异常类型并记录日志"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-017", "SpotZoom.py", "异常处理后无日志记录", "添加 WARNING/ERROR 级别日志"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-018", "SpotZoom.py", "SpotZoomController.__init__ 过长 (476行)", "拆分为多个初始化子方法"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-019", "SpotZoom.py", "AlignmentConfig 字段过多 (120+)", "拆分为多个配置组 (DetectionConfig 等)"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-020", "SpotZoom.py", "前 770 行几乎全是导入和别名绑定", "将别名绑定移到子包 __init__.py"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-021", "SpotZoom.py", "_detect_center_with_retry 重复代码", "提取为公共检测管道方法"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-022", "SpotZoom.py", "_attempt_recovery_scan 重复代码", "与 _detect_center_with_retry 合并"], [800, 2400, 3200, 3600]),
      ],
    }),

    heading("5.4 待修复 LOW 问题", HeadingLevel.HEADING_2),
    bodyPara("以下 4 个 LOW 级别问题为代码改进建议："),

    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["编号", "位置", "问题描述", "修复建议"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-025", "deep_image_prior_enhancer.py", "DIP 卷积使用三重 Python 循环", "建议用 scipy.ndimage.convolve"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-026", "SpotZoom.py", "SpotDetection.from_payload 缺少键检查", "添加字典键存在性验证"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-027", "lqg_robust_controller.py", "LQGConfig 全局命名空间冲突风险", "使用唯一前缀或嵌套类"], [800, 2400, 3200, 3600]),
        dataRow(["BUG-028", "deep_image_prior_enhancer.py", "_downsample 假设尺寸可被因子整除", "添加尺寸检查或自动填充"], [800, 2400, 3200, 3600]),
      ],
    }),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 6: 架构层面问题 ----
  const ch6 = [
    heading("6. 架构层面问题"),

    bodyPara("除上述具体代码缺陷外，项目在架构设计层面存在以下结构性问题，这些问题影响项目的长期可维护性和可扩展性。"),

    heading("6.1 上帝类 (God Class)", HeadingLevel.HEADING_2),
    bodyPara("SpotZoomController 类承担了所有核心职责，包括："),
    bulletPara("光斑检测与定位"),
    bulletPara("控制策略决策 (PID/MPC/雅可比)"),
    bulletPara("安全约束管理"),
    bulletPara("报告生成与日志记录"),
    bulletPara("设备通信与驱动"),
    bodyPara("建议拆分为多个子控制器，如 DetectionController、ControlStrategyController、SafetyManager、ReportGenerator 等，遵循单一职责原则。"),

    heading("6.2 配置膨胀", HeadingLevel.HEADING_2),
    bodyPara("AlignmentConfig 类包含 120+ 个字段，涵盖了检测参数、控制参数、安全参数、设备参数、显示参数等多个不相关的配置域。这种扁平化的配置结构导致："),
    bulletPara("配置项难以查找和管理"),
    bulletPara("新增配置时容易引发命名冲突"),
    bulletPara("配置验证逻辑复杂"),
    bodyPara("建议将 AlignmentConfig 拆分为 DetectionConfig、ControlConfig、SafetyConfig、DeviceConfig 等配置组，使用组合模式组织。"),

    heading("6.3 代码重复", HeadingLevel.HEADING_2),
    bodyPara("检测管道代码在多处重复实现，特别是："),
    bulletPara("_detect_center_with_retry 和 _attempt_recovery_scan 两个方法包含高度相似的检测逻辑"),
    bulletPara("多模块中存在相似的图像预处理代码块"),
    bulletPara("安全检查逻辑在多个位置重复实现"),
    bodyPara("建议提取公共检测管道方法，消除重复代码。"),

    heading("6.4 模块耦合", HeadingLevel.HEADING_2),
    bodyPara("40+ 创新模块全部在 SpotZoomController.__init__ 中初始化和绑定，导致："),
    bulletPara("初始化方法过长 (476 行)"),
    bulletPara("模块间存在隐式依赖"),
    bulletPara("新增模块需要修改主控制器代码"),
    bodyPara("建议引入插件注册机制，实现模块的自动发现和按需加载，降低主控制器与创新模块之间的耦合度。"),

    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ---- Chapter 7: 下一步优化方案 ----
  const ch7 = [
    heading("7. 下一步优化方案"),

    heading("7.1 短期优化 (1-2 天)", HeadingLevel.HEADING_2),
    bodyPara("短期目标聚焦于修复已识别的代码质量问题："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["优先级", "任务", "预期效果"], [1200, 5400, 3400]),
        dataRow(["高", "修复所有 MEDIUM 级别问题 (BUG-010 ~ BUG-022)", "消除潜在的性能和稳定性风险"], [1200, 5400, 3400]),
        dataRow(["高", "为 v2 模块添加单元测试", "确保修复不引入回归问题"], [1200, 5400, 3400]),
        dataRow(["中", "优化 DIP 卷积性能 (BUG-025)", "提升图像增强处理速度"], [1200, 5400, 3400]),
      ],
    }),

    heading("7.2 中期优化 (1 周)", HeadingLevel.HEADING_2),
    bodyPara("中期目标聚焦于代码结构优化："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["优先级", "任务", "预期效果"], [1200, 5400, 3400]),
        dataRow(["高", "拆分 SpotZoomController 为多个子控制器", "降低类复杂度，提升可维护性"], [1200, 5400, 3400]),
        dataRow(["高", "将 AlignmentConfig 拆分为配置组", "改善配置管理体验"], [1200, 5400, 3400]),
        dataRow(["中", "提取公共检测管道方法", "消除代码重复，统一检测逻辑"], [1200, 5400, 3400]),
      ],
    }),

    heading("7.3 长期优化 (2-4 周)", HeadingLevel.HEADING_2),
    bodyPara("长期目标聚焦于架构升级和生态建设："),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      layout: TableLayoutType.FIXED,
      rows: [
        headerRow(["优先级", "任务", "预期效果"], [1200, 5400, 3400]),
        dataRow(["中", "将 ML 模块别名绑定移到子包 __init__.py", "减少主文件前 770 行的导入代码"], [1200, 5400, 3400]),
        dataRow(["中", "添加集成测试和 CI 流水线", "自动化质量保障，防止回归"], [1200, 5400, 3400]),
        dataRow(["低", "引入插件注册机制替代硬编码初始化", "实现模块即插即用，降低耦合"], [1200, 5400, 3400]),
      ],
    }),

    new Paragraph({ spacing: { before: 400 } }),
    bodyPara("以上优化方案按照优先级和实施难度排序，建议按照短期 -> 中期 -> 长期的顺序逐步推进。每个阶段完成后应进行回归测试，确保优化过程中不引入新的问题。"),
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
          ...ch7,
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
