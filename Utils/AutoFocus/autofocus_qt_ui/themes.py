"""
autofocus_qt_ui 主题系统。

提供 dark / light 两套配色方案，以及对应的 QSS 样式表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Theme:
    """主题配色定义。"""

    name: str
    display_name: str

    # 背景色
    bg_primary: str       # 主背景
    bg_secondary: str     # 次级背景（面板、输入框）
    bg_tertiary: str      # 第三层背景（预览区、日志区）

    # 文字色
    text_primary: str
    text_secondary: str
    text_disabled: str

    # 强调色
    accent: str           # 主强调色（蓝色）
    accent_hover: str
    accent_pressed: str

    # 状态色
    success: str          # 成功/目标阈值
    warning: str          # 警告/触发阈值
    danger: str           # 错误/停止

    # 边框与分割线
    border: str
    divider: str

    # 图表颜色
    plot_bg: str
    plot_line: str
    plot_point: str
    plot_grid: str
    plot_text: str

    # ROI 框
    roi_border: str
    roi_text: str

    # 表格
    table_header_bg: str
    table_row_alt: str

    # 日志
    log_text: str

    def qss(self) -> str:
        """生成该主题对应的 Qt 样式表。"""
        return f"""
        QMainWindow {{ background-color: {self.bg_primary}; }}
        QWidget {{
            font-family: "Microsoft YaHei", "SimHei", sans-serif;
            font-size: 10pt;
            color: {self.text_primary};
        }}
        QGroupBox {{
            background-color: {self.bg_secondary};
            border: 1px solid {self.border};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: {self.accent};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            background-color: transparent;
        }}
        QPushButton {{
            background-color: {self.bg_secondary};
            border: 1px solid {self.border};
            border-radius: 4px;
            padding: 6px 14px;
            color: {self.text_primary};
        }}
        QPushButton:hover {{ background-color: {self._mix(self.bg_secondary, self.accent, 0.15)}; }}
        QPushButton:pressed {{ background-color: {self._mix(self.bg_secondary, self.accent, 0.30)}; }}
        QPushButton:disabled {{ background-color: {self.bg_primary}; color: {self.text_disabled}; }}
        QPushButton#primary {{
            background-color: {self.accent};
            color: {self._contrast_text(self.accent)};
            border: none;
        }}
        QPushButton#primary:hover {{ background-color: {self.accent_hover}; }}
        QPushButton#primary:pressed {{ background-color: {self.accent_pressed}; }}
        QPushButton#danger {{ background-color: {self.danger}; color: #ffffff; border: none; }}
        QPushButton#danger:hover {{ background-color: {self._mix(self.danger, "#000000", 0.15)}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {self.bg_secondary};
            border: 1px solid {self.border};
            border-radius: 4px;
            padding: 4px;
            color: {self.text_primary};
        }}
        QPlainTextEdit {{
            background-color: {self.bg_tertiary};
            border: 1px solid {self.border};
            color: {self.log_text};
            font-family: Consolas, monospace;
        }}
        QPlainTextEdit::selection {{
            background-color: {self.accent};
            color: {self._contrast_text(self.accent)};
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 1px solid {self.accent};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {self.bg_secondary};
            color: {self.text_primary};
            selection-background-color: {self.accent};
            selection-color: {self._contrast_text(self.accent)};
            border: 1px solid {self.border};
        }}
        QCheckBox {{ spacing: 6px; color: {self.text_primary}; }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {self.border};
            background-color: {self.bg_secondary};
        }}
        QCheckBox::indicator:checked {{
            background-color: {self.accent};
            border: 1px solid {self.accent};
            image: none;
        }}
        QCheckBox::indicator:unchecked:hover {{ border: 1px solid {self.accent}; }}
        QTableWidget::item:selected {{
            background-color: {self.accent};
            color: {self._contrast_text(self.accent)};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {self.border};
            margin: 4px 10px;
        }}
        QScrollBar:vertical {{
            background: {self.bg_primary};
            width: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {self.border};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {self.accent}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar:horizontal {{
            background: {self.bg_primary};
            height: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal {{
            background: {self.border};
            border-radius: 5px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {self.accent}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
        QProgressBar {{
            border: 1px solid {self.border};
            border-radius: 4px;
            text-align: center;
            color: {self.text_primary};
        }}
        QProgressBar::chunk {{ background-color: {self.accent}; }}
        QLabel#title {{ font-size: 14pt; font-weight: bold; color: {self.accent}; }}
        QLabel#status {{ color: {self.warning}; }}
        QTabWidget::pane {{ border: 1px solid {self.border}; background: {self.bg_secondary}; }}
        QTabBar::tab {{
            background: {self.bg_secondary};
            border: 1px solid {self.border};
            padding: 6px 14px;
            color: {self.text_primary};
        }}
        QTabBar::tab:selected {{ background: {self.accent}; color: {self._contrast_text(self.accent)}; }}
        QTabBar::tab:hover:!selected {{ background: {self._mix(self.bg_secondary, self.accent, 0.15)}; }}
        QScrollArea {{ border: none; background-color: {self.bg_primary}; }}
        QScrollArea > QWidget > QWidget {{ background-color: {self.bg_primary}; }}
        QSplitter::handle {{ background: {self.border}; }}
        QTableWidget {{
            background-color: {self.bg_tertiary};
            color: {self.text_primary};
            gridline-color: {self.border};
        }}
        QHeaderView::section {{
            background-color: {self.table_header_bg};
            color: {self.text_primary};
            padding: 4px;
            border: 1px solid {self.border};
        }}
        QMenuBar {{ background-color: {self.bg_primary}; color: {self.text_primary}; }}
        QMenuBar::item:selected {{ background-color: {self.accent}; color: {self._contrast_text(self.accent)}; }}
        QMenu {{ background-color: {self.bg_secondary}; color: {self.text_primary}; border: 1px solid {self.border}; }}
        QMenu::item:selected {{ background-color: {self.accent}; color: {self._contrast_text(self.accent)}; }}
        """

    @staticmethod
    def _mix(c1: str, c2: str, ratio: float) -> str:
        """混合两个十六进制颜色。"""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _contrast_text(bg: str) -> str:
        """根据背景色返回对比文字色（黑或白）。"""
        r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "#000000" if luminance > 0.5 else "#ffffff"


THEMES: Dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        display_name="深色主题",
        # 背景：更中性的深灰，去掉过重的蓝紫倾向
        bg_primary="#0f1115",
        bg_secondary="#181b21",
        bg_tertiary="#0a0c10",
        # 文字：更柔和的灰白，降低刺眼感
        text_primary="#e2e8f0",
        text_secondary="#94a3b8",
        text_disabled="#64748b",
        # 强调色：明亮的天蓝，对比度高且现代
        accent="#0ea5e9",
        accent_hover="#38bdf8",
        accent_pressed="#0284c7",
        # 状态色：更饱和、更易辨识
        success="#22c55e",
        warning="#f59e0b",
        danger="#ef4444",
        # 边框：低对比度但不消失
        border="#2d3748",
        divider="#1e293b",
        # 图表： cyan 线条在深色背景下更醒目
        plot_bg="#181b21",
        plot_line="#22d3ee",
        plot_point="#22d3ee",
        plot_grid="#334155",
        plot_text="#94a3b8",
        # ROI：与强调色一致但更亮
        roi_border="#38bdf8",
        roi_text="#38bdf8",
        # 表格
        table_header_bg="#1e293b",
        table_row_alt="#111418",
        # 日志：深色模式下保留轻微的终端绿感，但不过亮
        log_text="#86efac",
    ),
    "light": Theme(
        name="light",
        display_name="浅色主题",
        bg_primary="#f5f6fa",
        bg_secondary="#ffffff",
        bg_tertiary="#f0f1f5",
        text_primary="#2c3e50",
        text_secondary="#5a6a7a",
        text_disabled="#a0a8b0",
        accent="#2563eb",
        accent_hover="#3b82f6",
        accent_pressed="#1d4ed8",
        success="#16a34a",
        warning="#d97706",
        danger="#dc2626",
        border="#d1d5db",
        divider="#e5e7eb",
        plot_bg="#ffffff",
        plot_line="#2563eb",
        plot_point="#2563eb",
        plot_grid="#e5e7eb",
        plot_text="#6b7280",
        roi_border="#2563eb",
        roi_text="#2563eb",
        table_header_bg="#e5e7eb",
        table_row_alt="#f9fafb",
        # 日志：浅色模式用深灰绿
        log_text="#166534",
    ),
}


def get_theme(name: str = "light") -> Theme:
    """按名称获取主题，默认 light。"""
    return THEMES.get(name, THEMES["light"])


def available_themes() -> Dict[str, str]:
    """返回可用主题名称与显示名映射。"""
    return {name: theme.display_name for name, theme in THEMES.items()}
