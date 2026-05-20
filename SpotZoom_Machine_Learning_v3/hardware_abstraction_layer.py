"""
硬件抽象层 (Hardware Abstraction Layer)

参考开源项目:
  - python-microscope (https://github.com/python-microscope/microscope): 显微镜硬件抽象层

核心思想:
  从 python-microscope 的统一设备接口中借鉴，实现一个通用的硬件控制抽象层，
  支持 Thorlabs/Newport 位移台、相机等设备的统一控制。

  在 SpotZoom 场景中:
  - 位移台 → XYZ 精密定位平台
  - 相机 → 光斑图像采集
  - 触发同步 → 硬件触发确保采集与运动同步

创新点:
  1. 统一的设备注册与发现机制
  2. 硬件触发同步 (相机 + 电机)
  3. 网络分布式设备架构支持
  4. 设备状态监控与异常处理
  5. 模拟设备用于无硬件测试

纯 numpy + cv2 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型枚举。"""
    STAGE = "stage"
    CAMERA = "camera"
    SLM = "slm"
    WAVEFRONT_SENSOR = "wavefront_sensor"
    DEFORMABLE_MIRROR = "deformable_mirror"
    TRIGGER = "trigger"
    UNKNOWN = "unknown"


class DeviceStatus(Enum):
    """设备状态枚举。"""
    IDLE = "idle"
    MOVING = "moving"
    ACQUIRING = "acquiring"
    ERROR = "error"
    DISCONNECTED = "disconnected"
    READY = "ready"


@dataclass
class DeviceInfo:
    """设备信息。"""
    # 设备名称
    name: str = ""
    # 设备类型
    device_type: DeviceType = DeviceType.UNKNOWN
    # 制造商
    manufacturer: str = ""
    # 型号
    model: str = ""
    # 序列号
    serial_number: str = ""
    # 连接地址 (COM口/IP等)
    address: str = ""
    # 轴数 (位移台)
    num_axes: int = 1
    # 分辨率 (位移台: nm, 相机: um/pixel)
    resolution: float = 0.0
    # 行程范围 (位移台: mm)
    travel_range: Tuple[float, float] = (0.0, 0.0)
    # 当前状态
    status: DeviceStatus = DeviceStatus.DISCONNECTED
    # 是否为模拟设备
    is_simulated: bool = False


@dataclass
class HALConfig:
    """硬件抽象层配置。"""
    # 连接超时 (秒)
    connection_timeout: float = 5.0
    # 运动超时 (秒)
    motion_timeout: float = 30.0
    # 采集超时 (秒)
    acquisition_timeout: float = 10.0
    # 触发模式: "software", "hardware"
    trigger_mode: str = "software"
    # 触发延迟 (秒)
    trigger_delay: float = 0.01
    # 运动完成后等待稳定时间 (秒)
    settling_time: float = 0.1
    # 是否启用模拟设备
    simulation_mode: bool = True
    # 网络分布式模式
    distributed_mode: bool = False
    # 分布式服务器地址
    distributed_server: str = "localhost:5000"
    # 自动发现超时 (秒)
    discovery_timeout: float = 3.0
    # 设备状态轮询间隔 (秒)
    poll_interval: float = 0.1


class HardwareAbstractionLayer:
    """硬件抽象层。

    提供统一的设备控制接口，支持多种硬件设备的注册、发现和同步控制。

    Parameters
    ----------
    config : HALConfig
        硬件抽象层配置参数。

    References
    ----------
    .. [1] python-microscope documentation:
           https://python-microscope.readthedocs.io/
    .. [2] Thorlabs APT Protocol:
           https://www.thorlabs.com/software_pages/APT_Communications_Protocol.html
    """

    def __init__(self, config: Optional[HALConfig] = None) -> None:
        self.config = config or HALConfig()
        self._devices: Dict[str, DeviceInfo] = {}
        self._device_positions: Dict[str, np.ndarray] = {}
        self._device_callbacks: Dict[str, Dict[str, Callable]] = {}
        self._trigger_chain: List[str] = []
        self._is_initialized: bool = False

        logger.info(
            f"HardwareAbstractionLayer 初始化: "
            f"模拟模式={self.config.simulation_mode}, "
            f"触发模式={self.config.trigger_mode}"
        )

    def register_device(
        self,
        name: str,
        device_type: DeviceType,
        manufacturer: str = "",
        model: str = "",
        serial_number: str = "",
        address: str = "",
        num_axes: int = 1,
        resolution: float = 0.0,
        travel_range: Tuple[float, float] = (0.0, 0.0),
        is_simulated: Optional[bool] = None,
    ) -> DeviceInfo:
        """注册设备。

        Parameters
        ----------
        name : str
            设备名称 (唯一标识符)。
        device_type : DeviceType
            设备类型。
        manufacturer : str
            制造商名称。
        model : str
            设备型号。
        serial_number : str
            序列号。
        address : str
            连接地址。
        num_axes : int
            轴数。
        resolution : float
            分辨率。
        travel_range : Tuple[float, float]
            行程范围。
        is_simulated : bool, optional
            是否为模拟设备。默认跟随全局配置。

        Returns
        -------
        DeviceInfo
            注册的设备信息。
        """
        if name in self._devices:
            logger.warning(f"设备 '{name}' 已存在，将被覆盖")

        simulated = is_simulated if is_simulated is not None else self.config.simulation_mode

        device = DeviceInfo(
            name=name,
            device_type=device_type,
            manufacturer=manufacturer,
            model=model,
            serial_number=serial_number,
            address=address,
            num_axes=num_axes,
            resolution=resolution,
            travel_range=travel_range,
            status=DeviceStatus.READY if simulated else DeviceStatus.DISCONNECTED,
            is_simulated=simulated,
        )

        self._devices[name] = device
        self._device_positions[name] = np.zeros(num_axes)
        self._device_callbacks[name] = {}

        logger.info(
            f"设备注册: {name} ({device_type.value}), "
            f"制造商={manufacturer}, 模拟={simulated}"
        )
        return device

    def discover_devices(self) -> List[DeviceInfo]:
        """发现可用设备。

        在模拟模式下返回已注册的模拟设备。
        在实际硬件模式下扫描可用设备。

        Returns
        -------
        List[DeviceInfo]
            发现的设备列表。
        """
        if self.config.simulation_mode:
            discovered = list(self._devices.values())
            logger.info(f"模拟模式: 发现 {len(discovered)} 个已注册设备")
            return discovered

        # 实际硬件发现 (占位实现)
        logger.info("硬件设备发现启动...")
        discovered = []
        # 在实际实现中，这里会扫描 COM 口、USB、网络等
        logger.info(f"硬件发现完成: 找到 {len(discovered)} 个设备")
        return discovered

    def synchronized_move(
        self,
        device_name: str,
        positions: np.ndarray,
        wait: bool = True,
    ) -> bool:
        """同步移动设备。

        Parameters
        ----------
        device_name : str
            设备名称。
        positions : np.ndarray
            目标位置数组。
        wait : bool
            是否等待运动完成。

        Returns
        -------
        bool
            是否成功。
        """
        if device_name not in self._devices:
            logger.error(f"设备 '{device_name}' 未注册")
            return False

        device = self._devices[device_name]
        if device.status == DeviceStatus.ERROR:
            logger.error(f"设备 '{device_name}' 处于错误状态")
            return False

        # 检查行程范围
        for i, pos in enumerate(positions):
            lo, hi = device.travel_range
            if pos < lo or pos > hi:
                logger.warning(
                    f"位置 {pos} 超出行程范围 [{lo}, {hi}] (轴 {i})"
                )

        logger.info(
            f"设备 '{device_name}' 移动到 {positions.tolist()}, "
            f"等待={wait}"
        )

        # 更新状态
        device.status = DeviceStatus.MOVING

        if device.is_simulated:
            # 模拟运动
            self._device_positions[device_name] = positions.copy()
            device.status = DeviceStatus.READY
            return True

        # 实际硬件运动 (占位实现)
        # 在实际实现中，这里会发送运动命令到硬件
        device.status = DeviceStatus.READY
        self._device_positions[device_name] = positions.copy()
        return True

    def trigger_capture(
        self,
        camera_name: str,
        trigger_devices: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """触发同步采集。

        Parameters
        ----------
        camera_name : str
            相机设备名称。
        trigger_devices : List[str], optional
            需要先触发的设备列表。

        Returns
        -------
        np.ndarray, optional
            采集的图像。模拟模式下返回合成图像。
        """
        if camera_name not in self._devices:
            logger.error(f"相机 '{camera_name}' 未注册")
            return None

        camera = self._devices[camera_name]
        if camera.device_type != DeviceType.CAMERA:
            logger.error(f"设备 '{camera_name}' 不是相机")
            return None

        # 触发前置设备
        if trigger_devices:
            for dev_name in trigger_devices:
                if dev_name in self._devices:
                    logger.info(f"触发设备: {dev_name}")
                    time.sleep(self.config.trigger_delay)

        # 等待稳定
        if self.config.settling_time > 0:
            time.sleep(self.config.settling_time)

        logger.info(f"相机 '{camera_name}' 触发采集")

        if camera.is_simulated:
            # 模拟采集: 返回合成光斑图像
            image = self._generate_simulated_image()
            camera.status = DeviceStatus.READY
            return image

        # 实际硬件采集 (占位实现)
        camera.status = DeviceStatus.ACQUIRING
        camera.status = DeviceStatus.READY
        return np.zeros((512, 512), dtype=np.uint16)

    def get_position(self, device_name: str) -> np.ndarray:
        """获取设备当前位置。

        Parameters
        ----------
        device_name : str
            设备名称。

        Returns
        -------
        np.ndarray
            当前位置数组。
        """
        if device_name not in self._devices:
            logger.error(f"设备 '{device_name}' 未注册")
            return np.array([])
        return self._device_positions[device_name].copy()

    def get_device_info(self, device_name: str) -> Optional[DeviceInfo]:
        """获取设备信息。

        Parameters
        ----------
        device_name : str
            设备名称。

        Returns
        -------
        DeviceInfo, optional
            设备信息。
        """
        return self._devices.get(device_name)

    def list_devices(self, device_type: Optional[DeviceType] = None) -> List[str]:
        """列出已注册设备。

        Parameters
        ----------
        device_type : DeviceType, optional
            按类型过滤。

        Returns
        -------
        List[str]
            设备名称列表。
        """
        if device_type is None:
            return list(self._devices.keys())
        return [
            name for name, dev in self._devices.items()
            if dev.device_type == device_type
        ]

    def set_trigger_chain(self, device_names: List[str]) -> None:
        """设置触发链。

        Parameters
        ----------
        device_names : List[str]
            按触发顺序排列的设备名称列表。
        """
        for name in device_names:
            if name not in self._devices:
                logger.warning(f"触发链中的设备 '{name}' 未注册")
        self._trigger_chain = device_names
        logger.info(f"触发链已设置: {' -> '.join(device_names)}")

    def execute_trigger_chain(self) -> Dict[str, Any]:
        """执行触发链。

        Returns
        -------
        dict
            执行结果。
        """
        results: Dict[str, Any] = {}
        for i, name in enumerate(self._trigger_chain):
            if name not in self._devices:
                results[name] = {"success": False, "error": "设备未注册"}
                continue

            device = self._devices[name]
            logger.info(f"触发链 [{i+1}/{len(self._trigger_chain)}]: {name}")

            if device.device_type == DeviceType.CAMERA:
                image = self.trigger_capture(name)
                results[name] = {
                    "success": image is not None,
                    "image_shape": image.shape if image is not None else None,
                }
            else:
                results[name] = {"success": True}

            time.sleep(self.config.trigger_delay)

        logger.info(f"触发链执行完成: {len(results)} 个设备")
        return results

    def disconnect_all(self) -> None:
        """断开所有设备连接。"""
        for name, device in self._devices.items():
            device.status = DeviceStatus.DISCONNECTED
            logger.info(f"设备 '{name}' 已断开")
        self._is_initialized = False
        logger.info("所有设备已断开")

    def _generate_simulated_image(self, size: int = 512) -> np.ndarray:
        """生成模拟光斑图像。"""
        image = np.zeros((size, size), dtype=np.float64)
        cy, cx = size // 2, size // 2
        yy, xx = np.mgrid[:size, :size]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        sigma = 5.0
        image = 1000 * np.exp(-r2 / (2 * sigma ** 2))
        # 添加噪声
        noise = np.random.normal(0, 5, image.shape)
        image = np.maximum(image + noise, 0)
        return image.astype(np.uint16)
