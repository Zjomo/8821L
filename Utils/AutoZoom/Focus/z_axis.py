"""
Z 轴 (Newport 8742 Picomotor) 控制模块。

从 measurement_autofocus_shg_closed_loop.py 提取：
  - connect_z_controller()
  - setup_z_velocity()
  - move_z_relative()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import AutofocusConfig

logger = logging.getLogger(__name__)

# pylablib 延迟导入：避免触发 llvmlite DLL 加载失败（仅在连接时才导入）
Newport = None


def _import_newport():
    """懒加载 pylablib.devices.Newport，仅在需要连接电机时调用。"""
    global Newport
    if Newport is not None:
        return Newport
    try:
        from pylablib.devices import Newport as _Newport
        Newport = _Newport
        return Newport
    except ImportError:
        return None


class ZAxisController:
    """
    Newport 8742 Picomotor Z 轴控制器。

    用法:
        z = ZAxisController(cfg)
        z.connect()
        z.move_relative(100)   # 正向 100 步
        z.move_relative(-50)   # 反向 50 步
    """

    def __init__(self, cfg: AutofocusConfig):
        self.cfg = cfg
        self.controller: Optional[Newport.Picomotor8742] = None

    def _conn_kwargs(self) -> Dict[str, Any]:
        """从配置中提取连接参数，避免硬编码。"""
        return {
            # "conn": int(self.cfg.z_picomotor_conn),
            "conn": str(self.cfg.z_picomotor_conn),
            "backend": str(self.cfg.z_picomotor_backend or "auto"),
            "timeout": float(self.cfg.z_picomotor_timeout),
            "multiaddr": bool(self.cfg.z_picomotor_multiaddr),
            "scan": bool(self.cfg.z_picomotor_scan),
        }

    def connect(self) -> "ZAxisController":
        """
        连接 Newport 8742 控制器。
        如果已连接则直接返回。
        """
        if self.controller is not None:
            logger.info("[Z轴] 已连接")
            return self

        if _import_newport() is None:
            raise ImportError(
                "需要安装 pylablib 才能控制 Newport 8742：pip install pylablib"
            )

        if not self.cfg.z_enabled:
            raise RuntimeError("Z 轴补焦未启用 (z_enabled=False)")

        try:
            num_devices = Newport.get_usb_devices_number_picomotor()
        except Exception as e:
            raise RuntimeError(f"检测 Z 轴控制器失败: {e}") from e
        logger.info(f"[Z轴] 检测到 Newport Picomotor 数量：{num_devices}")
        if num_devices <= 0:
            raise RuntimeError("未检测到 Newport Picomotor 控制器")

        conn_kwargs = self._conn_kwargs()
        logger.info(f"[Z轴] 正在连接: {conn_kwargs}")
        try:
            self.controller = Newport.Picomotor8742(**conn_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"连接 Picomotor 控制器失败 (conn={conn_kwargs['conn']}): {e}\n"
                f"请检查：1) USB 是否插好；2) 控制器索引是否正确；"
                f"3) 是否有其他程序占用该控制器。"
            ) from e

        try:
            device_id = self.controller.get_id()
            axes = self.controller.axes()
            logger.info(f"[Z轴] 设备ID：{device_id}, 可用轴：{axes}")
        except Exception:
            pass

        self._setup_velocity()
        logger.info("[Z轴] 连接成功")
        return self

    def _setup_velocity(self):
        """设置 Z 轴速度和加速度。"""
        if self.controller is None:
            return
        try:
            axis = int(self.cfg.z_axis)
            speed = int(
                self.cfg.z_picomotor_velocity
                if self.cfg.z_picomotor_velocity is not None
                else self.cfg.z_speed
            )
            accel = int(
                self.cfg.z_picomotor_acceleration
                if self.cfg.z_picomotor_acceleration is not None
                else self.cfg.z_accel
            )
            self.controller.setup_velocity(
                axis=axis, speed=speed, accel=accel
            )
            logger.info(
                f"[Z轴] 速度/加速度: axis={axis}, speed={speed}, accel={accel}"
            )
        except Exception as e:
            logger.warning(f"[Z轴] 设置速度失败：{e}")

    def move_relative(self, signed_steps: int):
        """
        相对移动 Z 轴。

        参数
        ----------
        signed_steps : int
            正数 = 正向移动，负数 = 反向移动。
        """
        steps = int(round(signed_steps))
        if steps == 0:
            return

        if self.controller is None:
            self.connect()
        if self.controller is None:
            raise RuntimeError("Z 轴控制器未连接")

        axis = int(self.cfg.z_axis)
        logger.info(f"[Z轴] move_by axis={axis}, signed_steps={steps}")
        try:
            self.controller.move_by(axis=axis, steps=steps)
        except TypeError:
            # 兼容旧版 pylablib 参数签名
            self.controller.move_by(axis, steps)
        except Exception as e:
            raise RuntimeError(
                f"Z 轴移动失败: axis={axis}, steps={steps}, error={e}"
            ) from e

    def close(self) -> None:
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception as exc:
                logger.warning(f"[Z轴] 断开失败：{exc}")
            finally:
                self.controller = None

    @property
    def is_connected(self) -> bool:
        return self.controller is not None

    def check_available(self) -> tuple[bool, Optional[str]]:
        """
        检测 Z 轴控制器是否可用（不实际建立连接）。

        返回:
          - (True, None): 可用（z_enabled=False 也算可用，不会实际移动）
          - (False, error_msg): 不可用，返回错误信息
        """
        if not self.cfg.z_enabled:
            # Z 轴未启用，不会移动，不报错
            return True, None

        if _import_newport() is None:
            return False, "缺少依赖 pylablib: pip install pylablib"

        try:
            num_devices = Newport.get_usb_devices_number_picomotor()
            if num_devices <= 0:
                return False, "未检测到 Newport Picomotor 控制器，请检查USB连接"
            conn = int(self.cfg.z_picomotor_conn)
            if conn >= num_devices:
                return (
                    False,
                    f"控制器索引 {conn} 超出范围（共 {num_devices} 个设备），"
                    f"请调整 'Z 控制器索引' 为 0~{num_devices - 1}",
                )
            return True, None
        except Exception as e:
            return False, f"检测 Z 轴控制器失败: {e}"