from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE

# ============================================================
# 输出设置
# ============================================================

OUTPUT_FILE = "Scheme_C_MRC_Active_Laser_Beam_Stabilization.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# 页面尺寸，单位为英寸
SLIDE_W = 13.333
SLIDE_H = 7.5

# ============================================================
# 方案 C 配色
# ============================================================

NAVY = RGBColor(9, 34, 72)
DARK_BLUE = RGBColor(15, 66, 116)
BLUE = RGBColor(24, 120, 184)
CYAN = RGBColor(0, 166, 190)
TEAL = RGBColor(37, 166, 153)
GREEN = RGBColor(67, 174, 119)
ORANGE = RGBColor(235, 128, 68)
RED = RGBColor(202, 70, 82)
PURPLE = RGBColor(100, 87, 166)

WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(30, 34, 40)
DARK_GRAY = RGBColor(76, 86, 101)
MID_GRAY = RGBColor(137, 148, 163)
LIGHT_GRAY = RGBColor(224, 232, 240)
PALE_BLUE = RGBColor(239, 247, 253)
PALE_CYAN = RGBColor(232, 249, 247)
PALE_ORANGE = RGBColor(255, 245, 235)
PALE_RED = RGBColor(253, 241, 242)

FONT_CN = "Microsoft YaHei"
FONT_EN = "Aptos"

# ============================================================
# 基础绘图函数
# ============================================================

def rgb(color):
    return color


def set_background(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rectangle(
    slide, x, y, w, h, fill_color=WHITE, line_color=None,
    rounded=False, transparency=0
):
    shape_type = (
        MSO_SHAPE.ROUNDED_RECTANGLE
        if rounded
        else MSO_SHAPE.RECTANGLE
    )

    shape = slide.shapes.add_shape(
        shape_type,
        Inches(x), Inches(y), Inches(w), Inches(h)
    )

    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.fill.transparency = transparency

    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(0.8)

    return shape


def add_line(slide, x1, y1, x2, y2, color=BLUE, width=1.5):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1),
        Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color
    line.line.width = Pt(width)
    return line


def add_text(
    slide, text, x, y, w, h,
    size=16,
    color=BLACK,
    bold=False,
    font=FONT_CN,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    italic=False
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(w), Inches(h)
    )

    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = valign

    paragraph = tf.paragraphs[0]
    paragraph.alignment = align

    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color

    return box


def add_rich_text(
    slide, runs, x, y, w, h,
    size=16,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(w), Inches(h)
    )

    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = valign

    paragraph = tf.paragraphs[0]
    paragraph.alignment = align

    for item in runs:
        run = paragraph.add_run()
        run.text = item.get("text", "")
        run.font.name = item.get("font", FONT_CN)
        run.font.size = Pt(item.get("size", size))
        run.font.bold = item.get("bold", False)
        run.font.italic = item.get("italic", False)
        run.font.color.rgb = item.get("color", BLACK)

    return box


def add_title(slide, chinese_title, english_title, page_number):
    add_line(slide, 0.48, 0.48, 12.86, 0.48, NAVY, 1.0)

    add_rectangle(
        slide, 0.48, 0.68, 0.44, 0.42,
        fill_color=NAVY,
        line_color=NAVY,
        rounded=True
    )

    add_text(
        slide, f"{page_number:02d}",
        0.48, 0.77, 0.44, 0.16,
        size=11,
        color=WHITE,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide, chinese_title,
        1.05, 0.62, 7.5, 0.38,
        size=23,
        color=NAVY,
        bold=True
    )

    add_text(
        slide, english_title,
        1.05, 1.05, 7.5, 0.22,
        size=10,
        color=BLUE,
        font=FONT_EN
    )

    add_text(
        slide,
        "MRC-inspired optical stabilization",
        10.0, 0.74, 2.8, 0.18,
        size=8,
        color=MID_GRAY,
        font=FONT_EN,
        align=PP_ALIGN.RIGHT
    )


def add_footer(slide, page_number):
    add_rectangle(
        slide, 0.48, 7.08, 12.38, 0.035,
        fill_color=BLUE,
        line_color=BLUE
    )

    add_text(
        slide,
        "Group Meeting | Active Laser Beam Stabilization",
        0.52, 7.17, 5.8, 0.16,
        size=7.5,
        color=MID_GRAY,
        font=FONT_EN
    )

    add_text(
        slide,
        f"{page_number} / 12",
        11.95, 7.17, 0.85, 0.16,
        size=7.5,
        color=MID_GRAY,
        font=FONT_EN,
        align=PP_ALIGN.RIGHT
    )


def new_slide(chinese_title, english_title, page_number, bg=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, bg)
    add_title(slide, chinese_title, english_title, page_number)
    add_footer(slide, page_number)
    return slide


def add_card(
    slide, x, y, w, h,
    title, body="",
    accent=BLUE,
    fill=WHITE,
    title_size=14,
    body_size=10.5
):
    add_rectangle(
        slide, x, y, w, h,
        fill_color=fill,
        line_color=LIGHT_GRAY,
        rounded=True
    )

    add_rectangle(
        slide, x, y, 0.07, h,
        fill_color=accent,
        line_color=accent
    )

    add_text(
        slide,
        title,
        x + 0.22, y + 0.16, w - 0.34, 0.28,
        size=title_size,
        color=NAVY,
        bold=True
    )

    if body:
        add_text(
            slide,
            body,
            x + 0.22, y + 0.58, w - 0.36, h - 0.68,
            size=body_size,
            color=DARK_GRAY
        )


def add_bullet(
    slide, text, x, y, w,
    bullet_color=CYAN,
    size=12,
    text_color=BLACK
):
    add_rectangle(
        slide,
        x, y + 0.12, 0.075, 0.075,
        fill_color=bullet_color,
        line_color=bullet_color,
        rounded=True
    )

    add_text(
        slide,
        text,
        x + 0.17, y, w - 0.17, 0.32,
        size=size,
        color=text_color
    )


def add_metric(slide, value, label, x, y, w, accent=BLUE):
    add_rectangle(
        slide,
        x, y, w, 0.92,
        fill_color=PALE_BLUE,
        line_color=None,
        rounded=True
    )

    add_text(
        slide,
        value,
        x + 0.08, y + 0.12, w - 0.16, 0.33,
        size=22,
        color=accent,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide,
        label,
        x + 0.08, y + 0.57, w - 0.16, 0.18,
        size=9,
        color=DARK_GRAY,
        align=PP_ALIGN.CENTER
    )


def add_flow_box(slide, text, x, y, w, h, color):
    add_rectangle(
        slide,
        x, y, w, h,
        fill_color=color,
        line_color=color,
        rounded=True
    )

    add_text(
        slide,
        text,
        x + 0.04, y + 0.12, w - 0.08, h - 0.20,
        size=10,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE
    )


def add_chevron(slide, x, y, w, h, color=BLUE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.CHEVRON,
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_stage_label(slide, text, x, y, w, color=BLUE):
    add_rectangle(
        slide,
        x, y, w, 0.30,
        fill_color=PALE_BLUE,
        line_color=None,
        rounded=True
    )

    add_text(
        slide,
        text,
        x + 0.06, y + 0.06, w - 0.12, 0.16,
        size=9,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER
    )


# ============================================================
# 第 1 页：封面
# ============================================================

slide = prs.slides.add_slide(prs.slide_layouts[6])
set_background(slide, WHITE)

add_rectangle(
    slide,
    0, 0, 5.05, 7.5,
    fill_color=NAVY,
    line_color=NAVY
)

# 左侧装饰光束
add_line(slide, 0.2, 5.95, 4.85, 3.05, CYAN, 4.0)
add_line(slide, 0.2, 6.22, 4.85, 3.32, TEAL, 1.2)
add_line(slide, 2.2, 7.5, 5.05, 5.72, BLUE, 2.0)

add_text(
    slide,
    "主动激光束稳定系统",
    0.62, 1.20, 3.95, 0.72,
    size=28,
    color=WHITE,
    bold=True
)

add_text(
    slide,
    "Active Laser Beam Stabilization",
    0.65, 2.05, 3.85, 0.30,
    size=14,
    color=CYAN,
    bold=True,
    font=FONT_EN
)

add_text(
    slide,
    "基于双级观测与压电转镜闭环控制的\n四轴光束位置—方向稳定架构",
    0.65, 2.75, 3.90, 0.95,
    size=16,
    color=WHITE
)

add_text(
    slide,
    "组会汇报 | Literature-based system analysis",
    0.65, 5.90, 3.90, 0.25,
    size=10,
    color=RGBColor(192, 218, 234),
    font=FONT_EN
)

add_text(
    slide,
    "方案 C · 第三套视觉主题",
    0.65, 6.35, 3.90, 0.25,
    size=10,
    color=RGBColor(192, 218, 234)
)

# 右侧标题
add_text(
    slide,
    "从“光斑漂移”到“实时锁定”",
    5.85, 1.00, 6.45, 0.48,
    size=25,
    color=NAVY,
    bold=True
)

add_text(
    slide,
    "Position control + angular control",
    5.88, 1.55, 5.90, 0.25,
    size=12,
    color=BLUE,
    font=FONT_EN
)

# 右侧光路
add_line(slide, 6.05, 4.15, 12.15, 4.15, LIGHT_GRAY, 1.2)
add_line(slide, 6.20, 4.15, 7.55, 3.05, CYAN, 4.5)
add_line(slide, 7.55, 3.05, 9.15, 4.15, CYAN, 4.5)
add_line(slide, 9.15, 4.15, 10.95, 3.38, CYAN, 4.5)
add_line(slide, 10.95, 3.38, 12.15, 4.15, CYAN, 4.5)

cover_nodes = [
    (6.10, 3.83, "Laser", "光源"),
    (7.23, 2.70, "Mirror 1", "压电转镜 1"),
    (8.85, 3.82, "Mirror 2", "压电转镜 2"),
    (10.58, 3.07, "Detector 1", "位置探测"),
    (11.82, 3.83, "Detector 2", "角度探测"),
]

for x, y, en, cn in cover_nodes:
    add_rectangle(
        slide, x, y, 0.70, 0.48,
        fill_color=WHITE,
        line_color=BLUE,
        rounded=True
    )
    add_text(
        slide, en,
        x, y + 0.09, 0.70, 0.15,
        size=7.5,
        color=NAVY,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )
    add_text(
        slide, cn,
        x - 0.12, y + 0.56, 0.94, 0.25,
        size=8.2,
        color=DARK_GRAY,
        align=PP_ALIGN.CENTER
    )

add_rectangle(
    slide,
    6.20, 5.20, 5.65, 0.80,
    fill_color=PALE_CYAN,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "核心闭环：测量误差 → 控制器 → 压电镜修正 → 再测量",
    6.42, 5.47, 5.20, 0.25,
    size=13.5,
    color=NAVY,
    bold=True,
    align=PP_ALIGN.CENTER
)

add_text(
    slide,
    "Based on MRC Compact Laser Beam Stabilization documents",
    5.88, 6.72, 6.30, 0.20,
    size=8.5,
    color=MID_GRAY,
    font=FONT_EN
)


# ============================================================
# 第 2 页：研究背景
# ============================================================

slide = new_slide(
    "研究背景与应用需求",
    "Background & Significance",
    2
)

add_text(
    slide,
    "光束漂移并非单一位置问题，而是位置与角度的耦合失稳。",
    0.65, 1.58, 11.90, 0.42,
    size=21,
    color=NAVY,
    bold=True
)

add_card(
    slide, 0.65, 2.25, 3.55, 3.58,
    "典型扰动来源 / Disturbances",
    accent=ORANGE,
    fill=PALE_ORANGE
)

for i, text in enumerate([
    "机械振动与平台微扰",
    "热漂移与支架形变",
    "长光路中的角度放大",
    "显微镜目标平面偏移"
]):
    add_bullet(
        slide, text,
        0.95, 3.02 + i * 0.57, 2.80,
        bullet_color=ORANGE,
        size=11
    )

add_card(
    slide, 4.53, 2.25, 3.55, 3.58,
    "实验后果 / Consequences",
    accent=RED,
    fill=WHITE
)

for i, text in enumerate([
    "光斑位置不稳定",
    "成像与测量重复性下降",
    "长时间实验出现系统性漂移",
    "手动校准无法持续补偿"
]):
    add_bullet(
        slide, text,
        4.83, 3.02 + i * 0.57, 2.80,
        bullet_color=RED,
        size=11
    )

add_card(
    slide, 8.41, 2.25, 3.55, 3.58,
    "系统需求 / Requirements",
    accent=TEAL,
    fill=PALE_CYAN
)

for i, text in enumerate([
    "实时、连续、低延迟",
    "同时控制位置与方向",
    "可集成既有光路",
    "具备异常检测与恢复机制"
]):
    add_bullet(
        slide, text,
        8.71, 3.02 + i * 0.57, 2.80,
        bullet_color=TEAL,
        size=11
    )

add_rectangle(
    slide,
    1.60, 6.25, 10.15, 0.50,
    fill_color=NAVY,
    line_color=NAVY,
    rounded=True
)

add_text(
    slide,
    "关键转变：从“事后校正”转向“主动闭环稳定”",
    1.75, 6.38, 9.85, 0.18,
    size=14.5,
    color=WHITE,
    bold=True,
    align=PP_ALIGN.CENTER
)


# ============================================================
# 第 3 页：科学问题与假设
# ============================================================

slide = new_slide(
    "科学问题与研究假设",
    "Scientific Questions & Hypotheses",
    3
)

add_text(
    slide,
    "如何利用两个观测平面，将光束的横向位置与传播方向分离并同时稳定？",
    0.65, 1.55, 12.00, 0.58,
    size=20,
    color=NAVY,
    bold=True
)

questions = [
    (
        "Q1",
        "位置是否稳定？",
        "Detector 1 能否锁定光束在中间压电镜处的位置？",
        BLUE
    ),
    (
        "Q2",
        "方向是否稳定？",
        "Detector 2 是否能够感知远端目标平面的角度漂移？",
        CYAN
    ),
    (
        "Q3",
        "精度由什么决定？",
        "探测器分辨率、镜—探测器距离与机械稳定性如何共同作用？",
        TEAL
    ),
    (
        "Q4",
        "如何可靠启动？",
        "怎样避免方向编码错误、振荡与控制级之间的串扰？",
        ORANGE
    )
]

question_positions = [
    (0.70, 2.55),
    (6.88, 2.55),
    (0.70, 4.45),
    (6.88, 4.45)
]

for (qid, title, body, accent), (x, y) in zip(
    questions, question_positions
):
    add_rectangle(
        slide, x, y, 5.70, 1.38,
        fill_color=WHITE,
        line_color=LIGHT_GRAY,
        rounded=True
    )

    add_rectangle(
        slide,
        x + 0.18, y + 0.25, 0.62, 0.62,
        fill_color=accent,
        line_color=accent,
        rounded=True
    )

    add_text(
        slide, qid,
        x + 0.18, y + 0.44, 0.62, 0.15,
        size=12,
        color=WHITE,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide, title,
        x + 1.02, y + 0.21, 4.20, 0.25,
        size=15,
        color=NAVY,
        bold=True
    )

    add_text(
        slide, body,
        x + 1.02, y + 0.62, 4.35, 0.46,
        size=10.5,
        color=DARK_GRAY
    )

add_rectangle(
    slide,
    2.15, 6.20, 9.05, 0.50,
    fill_color=PALE_CYAN,
    line_color=None,
    rounded=True
)

add_rich_text(
    slide,
    [
        {
            "text": "核心假设 / Hypothesis: ",
            "color": NAVY,
            "bold": True
        },
        {
            "text": "双级观测与双级执行可以将 beam position 与 beam angle 转化为两个可控误差通道。",
            "color": DARK_GRAY
        }
    ],
    2.38, 6.34, 8.60, 0.18,
    size=11
)


# ============================================================
# 第 4 页：总体架构
# ============================================================

slide = new_slide(
    "系统总体架构与研究设计",
    "Overall System Architecture & Research Design",
    4
)

add_text(
    slide,
    "四轴系统 = 两个独立控制级 × 每级二维执行",
    0.65, 1.55, 7.00, 0.42,
    size=21,
    color=NAVY,
    bold=True
)

architecture = [
    ("Laser", "激光器", BLUE),
    ("Mirror 1", "转镜 1", CYAN),
    ("Mirror 2", "转镜 2", TEAL),
    ("Detector 1", "位置探测", GREEN),
    ("Target", "目标平面", PURPLE),
    ("Detector 2", "角度探测", ORANGE),
    ("Controller", "控制器", NAVY)
]

x_positions = [
    0.70, 2.35, 4.00, 5.65, 7.30, 8.95, 10.60
]

for i, ((en, cn, color), x) in enumerate(
    zip(architecture, x_positions)
):
    add_rectangle(
        slide,
        x, 2.45, 1.23, 0.72,
        fill_color=WHITE,
        line_color=color,
        rounded=True
    )

    add_text(
        slide, en,
        x + 0.05, 2.62, 1.13, 0.15,
        size=9.5,
        color=NAVY,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide, cn,
        x + 0.05, 2.88, 1.13, 0.16,
        size=8.2,
        color=DARK_GRAY,
        align=PP_ALIGN.CENTER
    )

    if i < len(architecture) - 1:
        add_chevron(
            slide,
            x + 1.28, 2.63, 0.27, 0.34,
            color=BLUE
        )

# 虚线反馈回路
add_line(slide, 11.20, 3.18, 11.20, 4.25, NAVY, 1.2)
add_line(slide, 11.20, 4.25, 2.95, 4.25, NAVY, 1.2)
add_line(slide, 2.95, 4.25, 2.95, 3.18, NAVY, 1.2)

add_text(
    slide,
    "position / angle feedback",
    5.50, 4.36, 2.40, 0.18,
    size=9,
    color=NAVY,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)

add_card(
    slide, 0.75, 5.05, 3.72, 1.25,
    "1. 观测 / Sensing",
    "从两个平面获取 x/y position error",
    accent=BLUE,
    fill=PALE_BLUE
)

add_card(
    slide, 4.80, 5.05, 3.72, 1.25,
    "2. 控制 / Control",
    "控制器根据误差计算镜面修正量",
    accent=TEAL,
    fill=PALE_CYAN
)

add_card(
    slide, 8.85, 5.05, 3.72, 1.25,
    "3. 执行 / Actuation",
    "压电转镜补偿角度与光斑偏移",
    accent=ORANGE,
    fill=PALE_ORANGE
)


# ============================================================
# 第 5 页：标准光路布局
# ============================================================

slide = new_slide(
    "标准四轴光路布局",
    "Standard 4-Axis Optical Layout",
    5
)

add_text(
    slide,
    "两个 detector 位于不同的观测位置：一个锁位置，一个锁方向。",
    0.65, 1.55, 11.80, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

add_rectangle(
    slide,
    0.75, 2.30, 11.85, 2.65,
    fill_color=RGBColor(247, 250, 252),
    line_color=LIGHT_GRAY,
    rounded=True
)

# 主光路
add_line(slide, 1.10, 3.70, 11.95, 3.70, CYAN, 4.0)
add_line(slide, 3.00, 3.70, 4.30, 2.95, CYAN, 4.0)
add_line(slide, 4.30, 2.95, 6.05, 3.70, CYAN, 4.0)
add_line(slide, 6.05, 3.70, 8.05, 3.05, CYAN, 4.0)
add_line(slide, 8.05, 3.05, 11.70, 3.70, CYAN, 4.0)

optical_components = [
    (1.05, 3.42, "Laser", "光源", BLUE),
    (2.68, 3.46, "Mirror 1", "压电镜 1\n靠近光源端", CYAN),
    (4.00, 2.68, "Mirror 2", "压电镜 2\n中间位置", TEAL),
    (5.55, 3.35, "Detector 1", "位置锁定", GREEN),
    (7.75, 2.76, "Target", "目标平面", PURPLE),
    (10.80, 3.35, "Detector 2", "角度锁定", ORANGE)
]

for x, y, en, cn, color in optical_components:
    add_rectangle(
        slide,
        x, y, 1.00, 0.52,
        fill_color=WHITE,
        line_color=color,
        rounded=True
    )

    add_text(
        slide, en,
        x, y + 0.10, 1.00, 0.15,
        size=8.5,
        color=NAVY,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide, cn,
        x - 0.15, y + 0.63, 1.30, 0.34,
        size=8.3,
        color=DARK_GRAY,
        align=PP_ALIGN.CENTER
    )

add_line(slide, 4.50, 4.55, 7.55, 4.55, MID_GRAY, 1.0)
add_text(
    slide,
    "recommended separation ≥ 0.5 m",
    5.15, 4.65, 2.55, 0.18,
    size=9,
    color=DARK_GRAY,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)

add_metric(slide, "2", "piezo mirrors", 1.10, 5.55, 2.30, CYAN)
add_metric(slide, "2", "position detectors", 3.75, 5.55, 2.30, GREEN)
add_metric(slide, "4", "controlled axes", 6.40, 5.55, 2.30, ORANGE)
add_metric(slide, "0.5 m+", "recommended distance", 9.05, 5.55, 2.30, BLUE)


# ============================================================
# 第 6 页：闭环控制
# ============================================================

slide = new_slide(
    "双级闭环控制原理",
    "Two-Stage Closed-Loop Control",
    6
)

add_text(
    slide,
    "误差不是直接“调相机画面”，而是反馈到压电转镜的二维控制量。",
    0.65, 1.55, 11.70, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

# Stage 1
add_rectangle(
    slide,
    0.75, 2.28, 5.70, 2.55,
    fill_color=PALE_BLUE,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "Stage 1 | 位置环 Position loop",
    1.05, 2.53, 4.90, 0.25,
    size=16,
    color=BLUE,
    bold=True
)

add_text(
    slide,
    "e₁ = [Δx₁, Δy₁]",
    1.05, 3.02, 2.50, 0.28,
    size=18,
    color=NAVY,
    bold=True,
    font=FONT_EN
)

add_flow_box(slide, "Detector 1", 1.05, 3.60, 1.35, 0.50, BLUE)
add_chevron(slide, 2.53, 3.68, 0.30, 0.32, BLUE)
add_flow_box(slide, "Controller", 2.95, 3.60, 1.45, 0.50, TEAL)
add_chevron(slide, 4.53, 3.68, 0.30, 0.32, BLUE)
add_flow_box(slide, "Mirror 2 X/Y", 4.90, 3.60, 1.42, 0.50, ORANGE)

add_text(
    slide,
    "锁定光束在中间转镜处的位置",
    1.05, 4.28, 4.75, 0.24,
    size=11,
    color=DARK_GRAY,
    align=PP_ALIGN.CENTER
)

# Stage 2
add_rectangle(
    slide,
    6.85, 2.28, 5.70, 2.55,
    fill_color=PALE_CYAN,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "Stage 2 | 方向环 Angular loop",
    7.15, 2.53, 4.90, 0.25,
    size=16,
    color=TEAL,
    bold=True
)

add_text(
    slide,
    "e₂ = [Δx₂, Δy₂]",
    7.15, 3.02, 2.50, 0.28,
    size=18,
    color=NAVY,
    bold=True,
    font=FONT_EN
)

add_flow_box(slide, "Detector 2", 7.15, 3.60, 1.35, 0.50, GREEN)
add_chevron(slide, 8.63, 3.68, 0.30, 0.32, BLUE)
add_flow_box(slide, "Controller", 9.05, 3.60, 1.45, 0.50, TEAL)
add_chevron(slide, 10.63, 3.68, 0.30, 0.32, BLUE)
add_flow_box(slide, "Mirror 1 X/Y", 11.00, 3.60, 1.42, 0.50, ORANGE)

add_text(
    slide,
    "抑制远端目标平面的角度变化",
    7.15, 4.28, 4.75, 0.24,
    size=11,
    color=DARK_GRAY,
    align=PP_ALIGN.CENTER
)

add_rectangle(
    slide,
    2.00, 5.35, 9.35, 0.78,
    fill_color=NAVY,
    line_color=NAVY,
    rounded=True
)

add_text(
    slide,
    "u = Kp · e + Ki · ∫e dt    |    初期先关闭耦合项，再进行矩阵解耦",
    2.22, 5.61, 8.90, 0.22,
    size=13.5,
    color=WHITE,
    bold=True,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)


# ============================================================
# 第 7 页：探测器与换算
# ============================================================

slide = new_slide(
    "探测器信号与位置换算",
    "Detector Readout & Position Conversion",
    7
)

add_text(
    slide,
    "探测器输出原始电压；通过光束直径与强度归一化得到实际位置。",
    0.65, 1.55, 11.80, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

# 4QD 面板
add_rectangle(
    slide,
    0.72, 2.22, 5.75, 3.70,
    fill_color=PALE_BLUE,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "4-QD detector",
    1.05, 2.53, 2.80, 0.25,
    size=17,
    color=BLUE,
    bold=True,
    font=FONT_EN
)

cx = 2.25
cy = 4.15

add_rectangle(
    slide,
    cx - 0.70, cy - 0.70, 1.40, 1.40,
    fill_color=WHITE,
    line_color=BLUE,
    rounded=True
)

add_line(slide, cx, cy - 0.70, cx, cy + 0.70, BLUE, 1.0)
add_line(slide, cx - 0.70, cy, cx + 0.70, cy, BLUE, 1.0)

add_rectangle(
    slide,
    cx - 0.22, cy - 0.22, 0.44, 0.44,
    fill_color=CYAN,
    line_color=CYAN,
    rounded=True
)

add_text(
    slide,
    "x / y",
    cx - 0.30, cy + 0.88, 0.60, 0.18,
    size=10,
    color=DARK_GRAY,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)

add_text(
    slide,
    "近中心近似 / Near-center approximation",
    3.35, 3.23, 2.65, 0.22,
    size=10.5,
    color=NAVY,
    bold=True
)

add_text(
    slide,
    "x[μm] = D[μm] · x[V] / πI[V]\n\ny[μm] = D[μm] · y[V] / πI[V]",
    3.35, 3.72, 2.65, 1.15,
    size=14,
    color=BLACK,
    font=FONT_EN
)

# PSD 面板
add_rectangle(
    slide,
    6.85, 2.22, 5.75, 3.70,
    fill_color=PALE_CYAN,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "PSD detector",
    7.18, 2.53, 2.80, 0.25,
    size=17,
    color=TEAL,
    bold=True,
    font=FONT_EN
)

add_rectangle(
    slide,
    7.30, 3.35, 1.55, 1.55,
    fill_color=WHITE,
    line_color=TEAL,
    rounded=True
)

add_text(
    slide,
    "PSD",
    7.30, 3.92, 1.55, 0.25,
    size=18,
    color=TEAL,
    bold=True,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)

add_text(
    slide,
    "近似线性 / Approximately linear",
    9.25, 3.23, 2.65, 0.22,
    size=10.5,
    color=NAVY,
    bold=True
)

add_text(
    slide,
    "x[μm] = x[mV] / (1.2 ± 0.03)\n\ny[μm] = y[mV] / (1.2 ± 0.03)",
    9.25, 3.72, 2.65, 1.15,
    size=14,
    color=BLACK,
    font=FONT_EN
)

add_rectangle(
    slide,
    1.40, 6.30, 10.50, 0.48,
    fill_color=NAVY,
    line_color=NAVY,
    rounded=True
)

add_text(
    slide,
    "工程重点：控制前先完成光斑居中、强度调节与方向编码确认。",
    1.65, 6.43, 10.00, 0.18,
    size=12,
    color=WHITE,
    bold=True,
    align=PP_ALIGN.CENTER
)


# ============================================================
# 第 8 页：精度与角度
# ============================================================

slide = new_slide(
    "位置精度与角度精度",
    "Position & Angular Accuracy",
    8
)

add_text(
    slide,
    "系统性能由 detector resolution、光路几何和机械稳定性共同决定。",
    0.65, 1.55, 11.80, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

# 图表
chart_data = ChartData()
chart_data.categories = ["small beam", "4 mm", "6 mm"]
chart_data.add_series(
    "Representative resolution [μm]",
    (0.10, 0.50, 0.90)
)

chart = slide.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(0.75), Inches(2.22),
    Inches(6.05), Inches(3.30),
    chart_data
).chart

chart.has_legend = False
chart.has_title = True
chart.chart_title.text_frame.text = (
    "Detector resolution: representative literature levels"
)

try:
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 1.2
    chart.value_axis.major_unit = 0.2
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.axis_title.text = "resolution [μm]"
    chart.category_axis.axis_title.text = "beam diameter condition"
except Exception:
    pass

series = chart.series[0]
series.format.fill.solid()
series.format.fill.fore_color.rgb = CYAN
series.format.line.color.rgb = CYAN

# 右侧说明
add_card(
    slide,
    7.15, 2.22, 5.35, 1.08,
    "文献结论 / Literature conclusion",
    "小光束条件下分辨率可优于 100 nm；4 mm 光束约 0.5 μm；6 mm 光束仍可达到亚微米量级。",
    accent=BLUE,
    fill=PALE_BLUE,
    body_size=10
)

add_card(
    slide,
    7.15, 3.55, 5.35, 1.08,
    "角度换算 / Angular conversion",
    "α ≈ Δx / L；当 Δx = 0.5 μm、L = 0.5 m 时，α ≈ 0.994 μrad。",
    accent=TEAL,
    fill=PALE_CYAN,
    body_size=10
)

add_card(
    slide,
    7.15, 4.88, 5.35, 1.08,
    "工程限制 / Practical limitation",
    "稳定支架、低热膨胀材料和合适的 detector—mirror 距离是长期精度的必要条件。",
    accent=ORANGE,
    fill=PALE_ORANGE,
    body_size=10
)

add_text(
    slide,
    "注：柱状图为依据文献表述制作的代表性量级示意，不替代原始标定曲线。",
    0.88, 5.78, 5.55, 0.28,
    size=8.5,
    color=MID_GRAY
)


# ============================================================
# 第 9 页：安装与调试
# ============================================================

slide = new_slide(
    "系统安装与启动调试流程",
    "Installation & Commissioning Workflow",
    9
)

add_text(
    slide,
    "调试原则：先机械稳定，再光束预对准，最后逐级闭环。",
    0.65, 1.55, 11.70, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

steps = [
    ("01", "稳定安装", "Stable mounts", BLUE),
    ("02", "连接线缆", "Cable connection", CYAN),
    ("03", "调节强度", "Intensity adjustment", TEAL),
    ("04", "开环预对准", "Open-loop alignment", GREEN),
    ("05", "确认方向", "Direction coding", ORANGE),
    ("06", "逐级调 P", "Stage-wise tuning", RED),
    ("07", "全四轴运行", "Full operation", PURPLE)
]

for i, (number, chinese, english, color) in enumerate(steps):
    column = i % 4
    row = i // 4

    x = 0.70 + column * 3.05
    y = 2.30 + row * 1.58

    add_rectangle(
        slide,
        x, y, 2.45, 1.02,
        fill_color=WHITE,
        line_color=LIGHT_GRAY,
        rounded=True
    )

    add_rectangle(
        slide,
        x + 0.18, y + 0.19, 0.57, 0.57,
        fill_color=color,
        line_color=color,
        rounded=True
    )

    add_text(
        slide,
        number,
        x + 0.18, y + 0.38, 0.57, 0.15,
        size=11,
        color=WHITE,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide,
        chinese,
        x + 0.90, y + 0.22, 1.30, 0.20,
        size=12,
        color=NAVY,
        bold=True
    )

    add_text(
        slide,
        english,
        x + 0.90, y + 0.53, 1.35, 0.16,
        size=8.5,
        color=MID_GRAY,
        font=FONT_EN
    )

    if i < len(steps) - 1 and column < 3:
        add_chevron(
            slide,
            x + 2.50, y + 0.35, 0.28, 0.30,
            color=MID_GRAY
        )

add_rectangle(
    slide,
    0.75, 5.95, 11.75, 0.75,
    fill_color=NAVY,
    line_color=NAVY,
    rounded=True
)

add_rich_text(
    slide,
    [
        {
            "text": "Failure check: ",
            "color": CYAN,
            "bold": True,
            "font": FONT_EN
        },
        {
            "text": "红色 Range LED → 方向编码错误；启动振荡 → 降低 P 或切换低带宽；信号过低/饱和 → 调整增益或滤光片。",
            "color": WHITE
        }
    ],
    1.00, 6.19, 11.25, 0.22,
    size=11
)


# ============================================================
# 第 10 页：故障诊断
# ============================================================

slide = new_slide(
    "失稳模式、诊断与优化策略",
    "Failure Modes & Optimization",
    10
)

add_text(
    slide,
    "将故障现象映射到可操作的诊断路径，是系统工程化的关键。",
    0.65, 1.55, 11.80, 0.40,
    size=20,
    color=NAVY,
    bold=True
)

headers = [
    "现象 / Symptom",
    "可能原因 / Cause",
    "优先措施 / Action",
    "控制目标"
]

column_x = [0.70, 3.45, 6.90, 11.05]
column_w = [2.75, 3.45, 4.15, 1.75]

for x, w, header in zip(column_x, column_w, headers):
    add_rectangle(
        slide,
        x, 2.22, w, 0.52,
        fill_color=NAVY,
        line_color=NAVY
    )

    add_text(
        slide,
        header,
        x + 0.06, 2.38, w - 0.12, 0.16,
        size=9.5,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER
    )

table_rows = [
    (
        "启动后快速跑向边界",
        "x/y direction coding reversed",
        "反转对应方向开关，重新测试",
        "不越界"
    ),
    (
        "闭环来回振荡",
        "P factor 过高 / 机械支撑不稳",
        "降低 P；必要时选择 low bandwidth",
        "稳定"
    ),
    (
        "信号饱和",
        "光强过高或滤光配置不当",
        "降低增益、改变滤光片或缩小 ROI",
        "不过曝"
    ),
    (
        "无法检测光斑",
        "光路未对准 / detector 偏离",
        "关闭控制级，重新进行开环预对准",
        "可捕获"
    ),
    (
        "两级相互影响",
        "控制通道混合或布局不合理",
        "分级调试，后续进行灵敏度矩阵标定",
        "低串扰"
    )
]

for row_index, row_data in enumerate(table_rows):
    y = 2.74 + row_index * 0.68
    fill = WHITE if row_index % 2 == 0 else PALE_BLUE

    for x, w, text in zip(column_x, column_w, row_data):
        add_rectangle(
            slide,
            x, y, w, 0.66,
            fill_color=fill,
            line_color=LIGHT_GRAY
        )

        text_align = (
            PP_ALIGN.CENTER
            if x == column_x[3]
            else PP_ALIGN.LEFT
        )

        add_text(
            slide,
            text,
            x + 0.09, y + 0.16, w - 0.18, 0.30,
            size=8.9,
            color=DARK_GRAY,
            align=text_align
        )

add_rectangle(
    slide,
    1.22, 6.45, 10.85, 0.40,
    fill_color=PALE_ORANGE,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "调参顺序建议：方向编码 → 机械稳定性 → P factor → bandwidth → 耦合矩阵",
    1.42, 6.56, 10.45, 0.16,
    size=10.5,
    color=ORANGE,
    bold=True,
    align=PP_ALIGN.CENTER
)


# ============================================================
# 第 11 页：创新点与显微镜迁移
# ============================================================

slide = new_slide(
    "创新点、讨论与显微镜场景迁移",
    "Innovation, Discussion & Microscope Integration",
    11
)

add_text(
    slide,
    "从标准自由空间布局出发，可迁移到显微镜中的“位置—角度”联合稳定。",
    0.65, 1.55, 11.80, 0.40,
    size=19,
    color=NAVY,
    bold=True
)

innovations = [
    (
        "01",
        "双观测平面",
        "同一束光在两个空间位置被测量，使位置误差与角度误差具备可分辨性。",
        BLUE
    ),
    (
        "02",
        "两级执行闭环",
        "前级锁定中间镜面位置，后级抑制远端目标平面的角度漂移。",
        TEAL
    ),
    (
        "03",
        "低延迟模拟控制",
        "文献强调模拟闭环可实现实时补偿，避免明显延迟与阶梯效应。",
        ORANGE
    )
]

for i, (number, title, body, accent) in enumerate(innovations):
    x = 0.75 + i * 4.15

    add_rectangle(
        slide,
        x, 2.30, 3.70, 1.80,
        fill_color=WHITE,
        line_color=LIGHT_GRAY,
        rounded=True
    )

    add_rectangle(
        slide,
        x + 0.22, 2.58, 0.58, 0.58,
        fill_color=accent,
        line_color=accent,
        rounded=True
    )

    add_text(
        slide,
        number,
        x + 0.22, 2.77, 0.58, 0.14,
        size=11,
        color=WHITE,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide,
        title,
        x + 0.98, 2.56, 2.25, 0.24,
        size=14.5,
        color=NAVY,
        bold=True
    )

    add_text(
        slide,
        body,
        x + 0.25, 3.30, 3.16, 0.54,
        size=9.8,
        color=DARK_GRAY
    )

# 显微镜迁移区域
add_rectangle(
    slide,
    0.75, 4.55, 7.15, 1.68,
    fill_color=PALE_BLUE,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "显微镜迁移建议 / Microscope integration",
    1.05, 4.82, 4.80, 0.24,
    size=15,
    color=BLUE,
    bold=True
)

add_line(slide, 1.20, 5.63, 6.95, 5.63, CYAN, 3.5)

microscope_components = [
    (1.35, "Laser", BLUE),
    (2.65, "M1", CYAN),
    (4.05, "M2", TEAL),
    (5.35, "Detector", GREEN),
    (6.55, "Sample", PURPLE)
]

for x, label, color in microscope_components:
    add_rectangle(
        slide,
        x, 5.37, 0.60, 0.46,
        fill_color=WHITE,
        line_color=color,
        rounded=True
    )

    add_text(
        slide,
        label,
        x, 5.53, 0.60, 0.14,
        size=8,
        color=NAVY,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

add_text(
    slide,
    "单平面：先锁相机/样品平面位置\n双平面：再引入远端 detector，联合稳定位置与方向",
    1.05, 6.00, 6.20, 0.36,
    size=9.5,
    color=DARK_GRAY
)

# 讨论区
add_rectangle(
    slide,
    8.25, 4.55, 4.25, 1.68,
    fill_color=PALE_ORANGE,
    line_color=None,
    rounded=True
)

add_text(
    slide,
    "讨论 / Discussion",
    8.55, 4.82, 3.50, 0.24,
    size=15,
    color=ORANGE,
    bold=True
)

add_bullet(
    slide,
    "文献主要描述系统布局与调试方法",
    8.55, 5.25, 3.50,
    bullet_color=ORANGE,
    size=9.5
)

add_bullet(
    slide,
    "缺少特定显微镜系统的实测闭环数据",
    8.55, 5.62, 3.50,
    bullet_color=ORANGE,
    size=9.5
)


# ============================================================
# 第 12 页：总结与展望
# ============================================================

slide = new_slide(
    "总结与展望",
    "Conclusions & Outlook",
    12,
    bg=PALE_BLUE
)

add_text(
    slide,
    "Take-home message",
    0.75, 1.55, 4.00, 0.28,
    size=13,
    color=BLUE,
    bold=True,
    font=FONT_EN
)

add_text(
    slide,
    "四轴主动稳定的核心，是把光束漂移转化为可测、可控、可验证的闭环误差。",
    0.75, 1.95, 11.45, 0.64,
    size=23,
    color=NAVY,
    bold=True
)

summary_items = [
    (
        "1",
        "位置与方向必须分开观测",
        "Detector 1 关注镜面处位置，Detector 2 关注远端方向。",
        BLUE
    ),
    (
        "2",
        "调试流程决定闭环可靠性",
        "机械稳定、开环预对准、方向编码和逐级调参缺一不可。",
        TEAL
    ),
    (
        "3",
        "显微镜应用需要二次验证",
        "应结合目标平面、工作距离、样品扰动和相机数据完成实测评估。",
        ORANGE
    )
]

for i, (number, title, body, accent) in enumerate(summary_items):
    y = 3.00 + i * 0.85

    add_rectangle(
        slide,
        0.85, y, 11.65, 0.65,
        fill_color=WHITE,
        line_color=LIGHT_GRAY,
        rounded=True
    )

    add_rectangle(
        slide,
        1.10, y + 0.14, 0.38, 0.38,
        fill_color=accent,
        line_color=accent,
        rounded=True
    )

    add_text(
        slide,
        number,
        1.10, y + 0.25, 0.38, 0.13,
        size=10,
        color=WHITE,
        bold=True,
        font=FONT_EN,
        align=PP_ALIGN.CENTER
    )

    add_text(
        slide,
        title,
        1.75, y + 0.14, 3.45, 0.22,
        size=13,
        color=NAVY,
        bold=True
    )

    add_text(
        slide,
        body,
        5.10, y + 0.17, 6.80, 0.20,
        size=10.2,
        color=DARK_GRAY
    )

add_rectangle(
    slide,
    0.85, 5.75, 11.65, 0.72,
    fill_color=NAVY,
    line_color=NAVY,
    rounded=True
)

add_rich_text(
    slide,
    [
        {
            "text": "Outlook: ",
            "color": CYAN,
            "bold": True,
            "font": FONT_EN
        },
        {
            "text": "完成双 detector 标定、4×4 灵敏度矩阵辨识、闭环带宽测试，并与显微镜成像质量建立定量关联。",
            "color": WHITE
        }
    ],
    1.15, 5.98, 11.05, 0.22,
    size=11.8
)

add_text(
    slide,
    "Thank you | Questions & Discussion",
    3.72, 6.70, 5.90, 0.28,
    size=15,
    color=BLUE,
    bold=True,
    font=FONT_EN,
    align=PP_ALIGN.CENTER
)


# ============================================================
# 图表格式
# ============================================================

for slide in prs.slides:
    for shape in slide.shapes:
        try:
            has_chart = shape.has_chart
        except Exception:
            has_chart = False

        if not has_chart:
            continue

        chart = shape.chart

        try:
            chart.chart_title.text_frame.paragraphs[0].font.size = Pt(10)
            chart.chart_title.text_frame.paragraphs[0].font.bold = True
            chart.chart_title.text_frame.paragraphs[0].font.color.rgb = NAVY
        except Exception:
            pass

        try:
            chart.category_axis.tick_labels.font.size = Pt(8)
            chart.category_axis.tick_labels.font.name = FONT_EN
            chart.category_axis.tick_labels.font.color.rgb = DARK_GRAY

            chart.value_axis.tick_labels.font.size = Pt(8)
            chart.value_axis.tick_labels.font.name = FONT_EN
            chart.value_axis.tick_labels.font.color.rgb = DARK_GRAY
        except Exception:
            pass


# ============================================================
# 保存 PPTX
# ============================================================

prs.save(OUTPUT_FILE)
print("PPTX generation completed.")
print("Output file:", OUTPUT_FILE)
