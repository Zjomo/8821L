"""
napari 适配器模块 (NapariAdapter)

基于 napari/napari 的多维图像查看器，适配SpotZoom的光斑可视化场景。

参考项目: https://github.com/napari/napari
核心能力: 多维图像浏览、插件系统、GPU渲染

创新点借鉴:
1. 图层系统 (Layer-based architecture)
2. 多维数据支持 (n-dimensional)
3. 交互式标注
4. 插件扩展机制

功能:
- 提供napari风格的图层管理接口
- 支持光斑检测结果的叠加显示
- 集成到SpotZoom可视化流程

依赖: numpy, opencv-python
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import Enum
import numpy as np
import cv2


class LayerType(Enum):
    """图层类型。"""
    IMAGE = "image"
    LABELS = "labels"
    POINTS = "points"
    SHAPES = "shapes"
    VECTORS = "vectors"


@dataclass
class Layer:
    """图层基类。"""
    name: str
    layer_type: LayerType
    visible: bool = True
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageLayer(Layer):
    """图像图层。"""
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    colormap: str = "gray"
    contrast_limits: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        self.layer_type = LayerType.IMAGE
        if self.contrast_limits is None and self.data.size > 0:
            self.contrast_limits = (float(self.data.min()), float(self.data.max()))


@dataclass
class PointsLayer(Layer):
    """点图层（用于光斑标记）。"""
    data: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 2))
    face_color: str = "red"
    edge_color: str = "black"
    size: float = 10.0
    symbol: str = "disc"

    def __post_init__(self):
        self.layer_type = LayerType.POINTS

    def add_point(self, x: float, y: float, properties: Optional[Dict] = None) -> None:
        """添加点。"""
        point = np.array([[x, y]])
        if self.data.size == 0:
            self.data = point
        else:
            self.data = np.vstack([self.data, point])

        if properties:
            if "properties" not in self.metadata:
                self.metadata["properties"] = []
            self.metadata["properties"].append(properties)


@dataclass
class ShapesLayer(Layer):
    """形状图层（用于光斑轮廓）。"""
    data: List[np.ndarray] = field(default_factory=list)
    face_color: str = "transparent"
    edge_color: str = "red"
    edge_width: float = 2.0

    def __post_init__(self):
        self.layer_type = LayerType.SHAPES

    def add_polygon(self, points: np.ndarray) -> None:
        """添加多边形。"""
        self.data.append(points)

    def add_ellipse(self, center: Tuple[float, float], radii: Tuple[float, float]) -> None:
        """添加椭圆。"""
        # 生成椭圆点
        theta = np.linspace(0, 2*np.pi, 50)
        x = center[0] + radii[0] * np.cos(theta)
        y = center[1] + radii[1] * np.sin(theta)
        self.data.append(np.column_stack([x, y]))


@dataclass
class LabelsLayer(Layer):
    """标签图层（用于分割掩码）。"""
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    num_colors: int = 100

    def __post_init__(self):
        self.layer_type = LayerType.LABELS


class NapariAdapter:
    """napari风格的图层管理适配器。

    借鉴napari的核心思想，实现灵活的图层管理：
    1. 图层叠加显示
    2. 独立属性控制
    3. 交互式操作支持

    Parameters
    ----------
    canvas_size : Tuple[int, int]
        画布大小。
    """

    def __init__(
        self,
        canvas_size: Tuple[int, int] = (800, 600),
    ):
        self.canvas_size = canvas_size
        self.layers: List[Layer] = []
        self._layer_callbacks: Dict[str, List[Callable]] = {}

    def add_layer(self, layer: Layer) -> Layer:
        """添加图层。

        Parameters
        ----------
        layer : Layer
            要添加的图层。

        Returns
        -------
        Layer
            添加的图层。
        """
        self.layers.append(layer)
        return layer

    def add_image(
        self,
        data: np.ndarray,
        name: str = "Image",
        colormap: str = "gray",
        opacity: float = 1.0,
    ) -> ImageLayer:
        """添加图像图层。

        Parameters
        ----------
        data : np.ndarray
            图像数据。
        name : str
            图层名称。
        colormap : str
            颜色映射。
        opacity : float
            不透明度。

        Returns
        -------
        ImageLayer
            图像图层。
        """
        layer = ImageLayer(
            name=name,
            data=data,
            colormap=colormap,
            opacity=opacity
        )
        return self.add_layer(layer)

    def add_points(
        self,
        name: str = "Points",
        face_color: str = "red",
        size: float = 10.0,
    ) -> PointsLayer:
        """添加点图层。

        Parameters
        ----------
        name : str
            图层名称。
        face_color : str
            点的颜色。
        size : float
            点的大小。

        Returns
        -------
        PointsLayer
            点图层。
        """
        layer = PointsLayer(
            name=name,
            face_color=face_color,
            size=size
        )
        return self.add_layer(layer)

    def add_shapes(
        self,
        name: str = "Shapes",
        edge_color: str = "red",
    ) -> ShapesLayer:
        """添加形状图层。

        Parameters
        ----------
        name : str
            图层名称。
        edge_color : str
            边框颜色。

        Returns
        -------
        ShapesLayer
            形状图层。
        """
        layer = ShapesLayer(
            name=name,
            edge_color=edge_color
        )
        return self.add_layer(layer)

    def add_labels(
        self,
        data: np.ndarray,
        name: str = "Labels",
    ) -> LabelsLayer:
        """添加标签图层。

        Parameters
        ----------
        data : np.ndarray
            标签数据。
        name : str
            图层名称。

        Returns
        -------
        LabelsLayer
            标签图层。
        """
        layer = LabelsLayer(
            name=name,
            data=data
        )
        return self.add_layer(layer)

    def remove_layer(self, name: str) -> bool:
        """移除图层。

        Parameters
        ----------
        name : str
            图层名称。

        Returns
        -------
        bool
            是否成功移除。
        """
        for i, layer in enumerate(self.layers):
            if layer.name == name:
                self.layers.pop(i)
                return True
        return False

    def get_layer(self, name: str) -> Optional[Layer]:
        """获取图层。

        Parameters
        ----------
        name : str
            图层名称。

        Returns
        -------
        Optional[Layer]
            图层对象，如果不存在则返回None。
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def move_layer(self, name: str, index: int) -> bool:
        """移动图层位置。

        Parameters
        ----------
        name : str
            图层名称。
        index : int
            目标位置。

        Returns
        -------
        bool
            是否成功移动。
        """
        layer = self.get_layer(name)
        if layer is None:
            return False

        self.layers.remove(layer)
        index = max(0, min(index, len(self.layers)))
        self.layers.insert(index, layer)
        return True

    def render(self) -> np.ndarray:
        """渲染所有图层。

        Returns
        -------
        np.ndarray
            渲染结果。
        """
        # 创建空白画布
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)

        # 按顺序渲染每个图层
        for layer in self.layers:
            if not layer.visible:
                continue

            canvas = self._render_layer(canvas, layer)

        return canvas

    def _render_layer(self, canvas: np.ndarray, layer: Layer) -> np.ndarray:
        """渲染单个图层。"""
        if layer.layer_type == LayerType.IMAGE:
            return self._render_image_layer(canvas, layer)
        elif layer.layer_type == LayerType.POINTS:
            return self._render_points_layer(canvas, layer)
        elif layer.layer_type == LayerType.SHAPES:
            return self._render_shapes_layer(canvas, layer)
        elif layer.layer_type == LayerType.LABELS:
            return self._render_labels_layer(canvas, layer)
        return canvas

    def _render_image_layer(self, canvas: np.ndarray, layer: ImageLayer) -> np.ndarray:
        """渲染图像图层。"""
        if layer.data.size == 0:
            return canvas

        # 调整图像大小以匹配画布
        data = layer.data
        if data.shape[:2] != canvas.shape[:2]:
            data = cv2.resize(data, (canvas.shape[1], canvas.shape[0]))

        # 应用颜色映射
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        elif data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        elif data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        # 混合
        alpha = layer.opacity
        return cv2.addWeighted(canvas, 1 - alpha, data.astype(np.uint8), alpha, 0)

    def _render_points_layer(self, canvas: np.ndarray, layer: PointsLayer) -> np.ndarray:
        """渲染点图层。"""
        if layer.data.size == 0:
            return canvas

        # 颜色映射
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        face_color = color_map.get(layer.face_color, (0, 0, 255))
        edge_color = color_map.get(layer.edge_color, (0, 0, 0))

        result = canvas.copy()
        for point in layer.data:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(result, (x, y), int(layer.size / 2), face_color, -1)
                cv2.circle(result, (x, y), int(layer.size / 2), edge_color, 1)

        # 混合
        alpha = layer.opacity
        return cv2.addWeighted(canvas, 1 - alpha, result, alpha, 0)

    def _render_shapes_layer(self, canvas: np.ndarray, layer: ShapesLayer) -> np.ndarray:
        """渲染形状图层。"""
        if not layer.data:
            return canvas

        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "transparent": None,
        }
        edge_color = color_map.get(layer.edge_color, (0, 0, 255))

        result = canvas.copy()
        for shape in layer.data:
            points = shape.astype(np.int32).reshape(-1, 1, 2)
            if edge_color:
                cv2.polylines(result, [points], True, edge_color, int(layer.edge_width))

        alpha = layer.opacity
        return cv2.addWeighted(canvas, 1 - alpha, result, alpha, 0)

    def _render_labels_layer(self, canvas: np.ndarray, layer: LabelsLayer) -> np.ndarray:
        """渲染标签图层。"""
        if layer.data.size == 0:
            return canvas

        # 为每个标签分配颜色
        labels = layer.data
        if labels.shape[:2] != canvas.shape[:2]:
            labels = cv2.resize(labels, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 生成彩色标签图
        colored = np.zeros_like(canvas)
        unique_labels = np.unique(labels)
        np.random.seed(42)  # 固定随机种子以获得一致的颜色

        for label in unique_labels:
            if label == 0:
                continue
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask = (labels == label)
            colored[mask] = color

        alpha = layer.opacity * 0.5  # 标签层通常更透明
        return cv2.addWeighted(canvas, 1 - alpha, colored, alpha, 0)

    def clear(self) -> None:
        """清除所有图层。"""
        self.layers.clear()

    def export_screenshot(self, filepath: str) -> bool:
        """导出截图。

        Parameters
        ----------
        filepath : str
            保存路径。

        Returns
        -------
        bool
            是否成功导出。
        """
        try:
            rendered = self.render()
            cv2.imwrite(filepath, rendered)
            return True
        except Exception:
            return False


class SpotDetectionVisualizer:
    """光斑检测结果可视化器。

    专门用于可视化光斑检测结果。
    """

    def __init__(self, canvas_size: Tuple[int, int] = (800, 600)):
        self.adapter = NapariAdapter(canvas_size)

    def visualize_detection(
        self,
        image: np.ndarray,
        spots: List[Dict[str, Any]],
        show_contours: bool = True,
        show_centers: bool = True,
    ) -> np.ndarray:
        """可视化检测结果。

        Parameters
        ----------
        image : np.ndarray
            原始图像。
        spots : List[Dict[str, Any]]
            光斑列表，每个光斑包含center、contour等信息。
        show_contours : bool
            是否显示轮廓。
        show_centers : bool
            是否显示中心点。

        Returns
        -------
        np.ndarray
            可视化结果。
        """
        self.adapter.clear()

        # 添加图像层
        self.adapter.add_image(image, name="Original")

        # 添加轮廓层
        if show_contours and spots:
            shapes_layer = self.adapter.add_shapes(name="Contours", edge_color="green")
            for spot in spots:
                if "contour" in spot:
                    shapes_layer.add_polygon(spot["contour"])
                elif "bbox" in spot:
                    x, y, w, h = spot["bbox"]
                    points = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ])
                    shapes_layer.add_polygon(points)

        # 添加中心点层
        if show_centers and spots:
            points_layer = self.adapter.add_points(name="Centers", face_color="red", size=8)
            for spot in spots:
                if "center" in spot:
                    cx, cy = spot["center"]
                    points_layer.add_point(cx, cy, properties=spot)

        return self.adapter.render()

    def visualize_tracking(
        self,
        image: np.ndarray,
        tracks: List[List[Tuple[float, float]]],
    ) -> np.ndarray:
        """可视化跟踪结果。

        Parameters
        ----------
        image : np.ndarray
            原始图像。
        tracks : List[List[Tuple[float, float]]]
            轨迹列表。

        Returns
        -------
        np.ndarray
            可视化结果。
        """
        self.adapter.clear()

        # 添加图像层
        self.adapter.add_image(image, name="Original")

        # 为每条轨迹创建形状层
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

        for i, track in enumerate(tracks):
            if len(track) < 2:
                continue

            color = colors[i % len(colors)]
            shapes_layer = self.adapter.add_shapes(name=f"Track_{i}", edge_color=color)

            # 将轨迹转换为线段
            track_array = np.array(track)
            shapes_layer.add_polygon(track_array)

            # 添加起点和终点标记
            points_layer = self.adapter.add_points(name=f"Points_{i}", face_color=color, size=6)
            for point in track:
                points_layer.add_point(point[0], point[1])

        return self.adapter.render()
