"""
Vizarr 适配器模块 (VizarrAdapter)

基于 hms-dbmi/vizarr 的多尺度Zarr图像在线浏览工具，适配SpotZoom的光斑可视化场景。

参考项目: https://github.com/hms-dbmi/vizarr
论文: Manz et al. (2022) "Viv: multiscale visualization of high-resolution multiplexed bioimaging data on the web"

创新点借鉴:
1. GPU加速渲染 (基于Viv)
2. 纯客户端Zarr访问
3. 多尺度图像金字塔
4. OME-NGFF格式支持

功能:
- 提供Vizarr风格的多尺度图像查看接口
- 支持Zarr格式光斑数据的高效加载
- 集成到SpotZoom可视化流程

依赖: numpy, opencv-python (可选: zarr)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import cv2


@dataclass
class MultiscaleImage:
    """多尺度图像表示。"""
    base_image: np.ndarray  # 原始图像
    pyramid: List[np.ndarray]  # 图像金字塔
    scales: List[Tuple[float, float]]  # 每层的尺度因子
    tile_size: Tuple[int, int]  # 瓦片大小


@dataclass
class TileRequest:
    """瓦片请求。"""
    level: int  # 金字塔层级
    x: int  # 瓦片X坐标
    y: int  # 瓦片Y坐标
    z: Optional[int] = None  # Z切片（3D数据）


@dataclass
class TileData:
    """瓦片数据。"""
    data: np.ndarray  # 瓦片图像数据
    level: int
    coordinates: Tuple[int, int]


class VizarrAdapter:
    """Vizarr风格的多尺度图像适配器。

    借鉴Vizarr的核心思想，实现高效的多尺度光斑图像管理：
    1. 图像金字塔构建
    2. 按需瓦片加载
    3. GPU加速渲染准备
    4. 延迟加载策略

    Parameters
    ----------
    tile_size : Tuple[int, int]
        瓦片大小 (width, height)。
    max_levels : int
        最大金字塔层级数。
    downsample_factor : float
        每层下采样因子。
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (256, 256),
        max_levels: int = 5,
        downsample_factor: float = 2.0,
    ):
        self.tile_size = tile_size
        self.max_levels = max_levels
        self.downsample_factor = downsample_factor
        self._zarr_available = self._check_zarr()

    def _check_zarr(self) -> bool:
        """检查 zarr 库是否可用。"""
        try:
            import zarr
            return True
        except ImportError:
            return False

    def build_pyramid(
        self,
        image: np.ndarray,
    ) -> MultiscaleImage:
        """构建图像金字塔。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        MultiscaleImage
            多尺度图像对象。
        """
        pyramid = [image]
        scales = [(1.0, 1.0)]
        current = image.copy()

        for level in range(1, self.max_levels):
            # 检查是否还能继续下采样
            h, w = current.shape[:2]
            new_h = int(h / self.downsample_factor)
            new_w = int(w / self.downsample_factor)

            if new_h < self.tile_size[1] or new_w < self.tile_size[0]:
                break

            # 下采样
            current = cv2.resize(
                current,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
            pyramid.append(current)
            scales.append((
                scales[0][0] * (self.downsample_factor ** level),
                scales[0][1] * (self.downsample_factor ** level)
            ))

        return MultiscaleImage(
            base_image=image,
            pyramid=pyramid,
            scales=scales,
            tile_size=self.tile_size
        )

    def get_tile(
        self,
        multiscale: MultiscaleImage,
        request: TileRequest,
    ) -> Optional[TileData]:
        """获取指定瓦片数据。

        Parameters
        ----------
        multiscale : MultiscaleImage
            多尺度图像对象。
        request : TileRequest
            瓦片请求。

        Returns
        -------
        Optional[TileData]
            瓦片数据，如果超出范围则返回None。
        """
        if request.level >= len(multiscale.pyramid):
            return None

        image = multiscale.pyramid[request.level]
        tile_h, tile_w = self.tile_size

        # 计算瓦片在图像中的位置
        x_start = request.x * tile_w
        y_start = request.y * tile_h
        x_end = min(x_start + tile_w, image.shape[1])
        y_end = min(y_start + tile_h, image.shape[0])

        # 检查边界
        if x_start >= image.shape[1] or y_start >= image.shape[0]:
            return None

        # 提取瓦片
        tile = image[y_start:y_end, x_start:x_end]

        # 如果瓦片小于标准大小，进行填充
        if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
            if len(tile.shape) == 3:
                padded = np.zeros((tile_h, tile_w, tile.shape[2]), dtype=tile.dtype)
            else:
                padded = np.zeros((tile_h, tile_w), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded

        return TileData(
            data=tile,
            level=request.level,
            coordinates=(request.x, request.y)
        )

    def get_visible_tiles(
        self,
        multiscale: MultiscaleImage,
        viewport: Tuple[int, int, int, int],  # (x, y, width, height)
        level: int = 0,
    ) -> List[TileRequest]:
        """获取视口内所有瓦片请求。

        Parameters
        ----------
        multiscale : MultiscaleImage
            多尺度图像对象。
        viewport : Tuple[int, int, int, int]
            视口位置和大小 (x, y, width, height)。
        level : int
            金字塔层级。

        Returns
        -------
        List[TileRequest]
            瓦片请求列表。
        """
        if level >= len(multiscale.pyramid):
            level = len(multiscale.pyramid) - 1

        image = multiscale.pyramid[level]
        tile_h, tile_w = self.tile_size

        # 考虑层级缩放
        scale = multiscale.scales[level]
        vx, vy, vw, vh = viewport

        # 计算视口覆盖的瓦片范围
        start_x = max(0, vx // tile_w)
        start_y = max(0, vy // tile_h)
        end_x = min((vx + vw) // tile_w + 1, (image.shape[1] + tile_w - 1) // tile_w)
        end_y = min((vy + vh) // tile_h + 1, (image.shape[0] + tile_h - 1) // tile_h)

        requests = []
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                requests.append(TileRequest(level=level, x=x, y=y))

        return requests

    def select_optimal_level(
        self,
        multiscale: MultiscaleImage,
        viewport_size: Tuple[int, int],
        image_display_size: Tuple[int, int],
    ) -> int:
        """选择最优金字塔层级。

        Parameters
        ----------
        multiscale : MultiscaleImage
            多尺度图像对象。
        viewport_size : Tuple[int, int]
            视口大小。
        image_display_size : Tuple[int, int]
            图像显示大小。

        Returns
        -------
        int
            最优层级索引。
        """
        base_image = multiscale.base_image
        scale_x = image_display_size[0] / base_image.shape[1]
        scale_y = image_display_size[1] / base_image.shape[0]
        display_scale = min(scale_x, scale_y)

        # 找到最接近但不超过显示尺度的层级
        best_level = 0
        for level, scale in enumerate(multiscale.scales):
            level_scale = 1.0 / scale[0]
            if level_scale <= display_scale * 1.5:  # 允许一些上采样
                best_level = level
            else:
                break

        return best_level

    def export_to_zarr(
        self,
        multiscale: MultiscaleImage,
        output_path: str,
    ) -> bool:
        """导出为Zarr格式。

        Parameters
        ----------
        multiscale : MultiscaleImage
            多尺度图像对象。
        output_path : str
            输出路径。

        Returns
        -------
        bool
            是否成功导出。
        """
        if not self._zarr_available:
            return False

        try:
            import zarr

            # 创建Zarr存储
            store = zarr.DirectoryStore(output_path)
            root = zarr.group(store=store, overwrite=True)

            # 保存每层金字塔
            for level, image in enumerate(multiscale.pyramid):
                level_group = root.create_group(f"level_{level}")
                level_group.create_dataset(
                    "data",
                    data=image,
                    chunks=self.tile_size + (image.shape[2],) if len(image.shape) == 3 else self.tile_size,
                    compression="blosc"
                )
                level_group.attrs["scale"] = multiscale.scales[level]

            # 保存元数据
            root.attrs["tile_size"] = self.tile_size
            root.attrs["max_levels"] = len(multiscale.pyramid)
            root.attrs["downsample_factor"] = self.downsample_factor

            return True
        except Exception:
            return False


class GPUAcceleratedRenderer:
    """GPU加速渲染器（基于Viv的设计理念）。

    准备数据以供GPU渲染使用。
    """

    def __init__(self):
        self._cuda_available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """检查CUDA是否可用。"""
        try:
            import cv2
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def prepare_for_gpu(
        self,
        tile: np.ndarray,
    ) -> np.ndarray:
        """准备瓦片数据以供GPU渲染。

        Parameters
        ----------
        tile : np.ndarray
            输入瓦片。

        Returns
        -------
        np.ndarray
            处理后的数据。
        """
        # 确保数据类型适合GPU处理
        if tile.dtype != np.uint8:
            tile = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 转换为RGBA格式（如果需要）
        if len(tile.shape) == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGBA)
        elif tile.shape[2] == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGBA)

        return tile

    def apply_colormap(
        self,
        tile: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """应用颜色映射。

        Parameters
        ----------
        tile : np.ndarray
            输入瓦片（单通道）。
        colormap : int
            OpenCV颜色映射。

        Returns
        -------
        np.ndarray
            彩色图像。
        """
        if len(tile.shape) == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        return cv2.applyColorMap(tile, colormap)


class SpotVizarrViewer:
    """光斑专用的Vizarr风格查看器。

    集成多尺度图像管理与GPU加速渲染。
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (256, 256),
        max_levels: int = 5,
    ):
        self.adapter = VizarrAdapter(tile_size, max_levels)
        self.renderer = GPUAcceleratedRenderer()
        self._multiscale_cache: Dict[str, MultiscaleImage] = {}

    def load_image(
        self,
        image_id: str,
        image: np.ndarray,
    ) -> MultiscaleImage:
        """加载图像并构建金字塔。

        Parameters
        ----------
        image_id : str
            图像标识符。
        image : np.ndarray
            输入图像。

        Returns
        -------
        MultiscaleImage
            多尺度图像对象。
        """
        multiscale = self.adapter.build_pyramid(image)
        self._multiscale_cache[image_id] = multiscale
        return multiscale

    def render_viewport(
        self,
        image_id: str,
        viewport: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """渲染视口内容。

        Parameters
        ----------
        image_id : str
            图像标识符。
        viewport : Tuple[int, int, int, int]
            视口位置和大小。
        target_size : Tuple[int, int]
            目标输出大小。

        Returns
        -------
        Optional[np.ndarray]
            渲染结果。
        """
        if image_id not in self._multiscale_cache:
            return None

        multiscale = self._multiscale_cache[image_id]

        # 选择最优层级
        level = self.adapter.select_optimal_level(
            multiscale,
            (viewport[2], viewport[3]),
            target_size
        )

        # 获取可见瓦片
        tile_requests = self.adapter.get_visible_tiles(multiscale, viewport, level)

        # 渲染瓦片（简化版：直接返回对应层级的裁剪）
        image = multiscale.pyramid[level]
        x, y, w, h = viewport

        # 缩放视口坐标到当前层级
        scale = multiscale.scales[level]
        x = int(x / scale[0])
        y = int(y / scale[1])
        w = int(w / scale[0])
        h = int(h / scale[1])

        # 裁剪
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        if w <= 0 or h <= 0:
            return None

        result = image[y:y+h, x:x+w]

        # 调整到目标大小
        if result.shape[1] != target_size[0] or result.shape[0] != target_size[1]:
            result = cv2.resize(result, target_size, interpolation=cv2.INTER_LINEAR)

        return result

    def clear_cache(self) -> None:
        """清除缓存。"""
        self._multiscale_cache.clear()
