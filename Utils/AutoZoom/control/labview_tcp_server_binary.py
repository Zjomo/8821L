# logic/labview_tcp_server_binary.py
from __future__ import annotations

import csv
import socket
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

import numpy as np


LogCallback = Optional[Callable[[str], None]]


def save_values_to_csv(
    values: List[float],
    index: int,
    output_dir: str | Path = "labview_csv_output",
) -> str:
    """
    保存一次 LabVIEW 返回的光谱数据。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"measure_{index:03d}_{timestamp}.csv"

    with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["measure_index", index])
        writer.writerow(["timestamp", timestamp])
        writer.writerow(["num_points", len(values)])
        writer.writerow([])
        writer.writerow(["point_index", "value"])

        for i, value in enumerate(values):
            writer.writerow([i, value])

    return str(filepath)


class TCPBufferReader:
    """
    TCP 缓冲读取器。

    作用：
        1. 可读取 READY 这种文本行；
        2. 可精确读取二进制 header / payload；
        3. 避免 recv_line 多读到二进制数据后丢失。
    """

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.buffer = b""

    def recv_line(self) -> str:
        """
        读取一行文本消息。

        LabVIEW 端发送 READY 时，建议发送：
            READY\\n
        """
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                return line.decode("utf-8", errors="ignore").strip()

            chunk = self.conn.recv(4096)
            if not chunk:
                raise ConnectionError("LabVIEW 已断开 TCP 连接")

            self.buffer += chunk

    def recv_exactly(self, n: int) -> bytes:
        """
        精确读取 n 个字节。

        TCP 是字节流，不保证一次 recv 就能收到完整数据。
        所以这里必须循环读取，直到读满 n 个字节。
        """
        if n <= 0:
            return b""

        chunks: List[bytes] = []
        received = 0

        # 先消耗缓冲区中已有的数据
        if self.buffer:
            take = min(n, len(self.buffer))
            chunks.append(self.buffer[:take])
            self.buffer = self.buffer[take:]
            received += take

        while received < n:
            chunk = self.conn.recv(n - received)
            if not chunk:
                raise ConnectionError(
                    f"LabVIEW 连接断开：需要 {n} 字节，实际只收到 {received} 字节"
                )

            chunks.append(chunk)
            received += len(chunk)

        return b"".join(chunks)


class LabVIEWTCPServer:
    """
    Python 作为 TCP Server，LabVIEW 作为 TCP Client。

    当前二进制协议：

        1. Python -> LabVIEW:
            MEASURE

        2. LabVIEW -> Python:
            4 字节 header + payload

    其中：
        header:
            U32 类型的 payload 字节长度，
            由 LabVIEW 第二个“平化至字符串”生成。

        payload:
            DBL 光谱数组的二进制数据，
            由 LabVIEW 第一个“平化至字符串”生成。

    如果光谱数组是 1024 个 DBL：
        payload_len = 1024 * 8 = 8192
        总发送长度 = 4 + 8192 = 8196
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 65432,
        output_dir: str | Path = "labview_csv_output",
        on_log: LogCallback = None,
        socket_timeout_s: float = 30.0,
        expected_points: int = 1024,
        send_newline: bool = False,
        payload_byte_order: str = "big",
        auto_detect_header_byte_order: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.output_dir = Path(output_dir)
        self.on_log = on_log
        self.socket_timeout_s = float(socket_timeout_s)

        self.expected_points = int(expected_points)
        self.expected_payload_len = self.expected_points * 8

        # 你的 LabVIEW 左下角如果是直接判断字符串 == "MEASURE"，
        # 这里必须保持 False。
        # 如果 LabVIEW 端 TCP Read 后加了“删除空白字符”，可以改 True。
        self.send_newline = bool(send_newline)

        # 如果 LabVIEW 第一个“平化至字符串”的字节顺序设为“大端/网络字节顺序”，这里用 big。
        # 如果你没有设置字节顺序，Python 收到数值异常时可尝试 little。
        self.payload_byte_order = payload_byte_order

        # header 自动判断大端/小端。
        # 例如 header 原始字节是 00 00 20 00，则大端解释为 8192。
        self.auto_detect_header_byte_order = bool(auto_detect_header_byte_order)

        self.server_socket: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr = None
        self.reader: Optional[TCPBufferReader] = None

        self.is_server_running = False
        self.is_connected = False
        self.is_ready = False

        self.measure_count = 0

        self._accept_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    # ============================================================
    # 日志
    # ============================================================

    def log(self, msg: str) -> None:
        if self.on_log is not None:
            self.on_log(msg)
        else:
            print(msg)

    # ============================================================
    # Server 启动
    # ============================================================

    def start_server_async(self) -> None:
        """
        GUI 中调用这个方法，不阻塞界面。
        """
        with self._lock:
            if self.is_server_running:
                self.log("[LabVIEWTCP] TCP Server 已经在运行")
                return

            self._accept_thread = threading.Thread(
                target=self.start_server_blocking,
                daemon=True,
            )
            self._accept_thread.start()

    def start_server_blocking(self) -> None:
        """
        控制台测试时可以直接调用。
        GUI 中不要直接调用。
        """
        try:
            with self._lock:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(1)

                self.is_server_running = True
                self.is_connected = False
                self.is_ready = False

            self.log(f"[LabVIEWTCP] TCP Server 已启动: {self.host}:{self.port}")
            self.log("[LabVIEWTCP] 等待 LabVIEW 连接...")

            conn, addr = self.server_socket.accept()
            conn.settimeout(self.socket_timeout_s)

            # 降低小包延迟，适合命令-响应式采集
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            with self._lock:
                self.conn = conn
                self.addr = addr
                self.reader = TCPBufferReader(conn)
                self.is_connected = True

            self.log(f"[LabVIEWTCP] LabVIEW 已连接: {addr}")

        except Exception as e:
            self.log(f"[LabVIEWTCP] TCP Server 启动或连接失败: {e}")
            self.close()

    # ============================================================
    # READY 等待
    # ============================================================

    def wait_for_ready(self) -> Dict[str, Any]:
        """
        等待 LabVIEW 发送 READY。

        你的 LabVIEW 图中左上角看起来有 READY\\n 发送逻辑，
        所以这里按一行文本读取 READY。
        """
        try:
            self._ensure_connected()

            self.log("[LabVIEWTCP] 等待 LabVIEW Setup 完成并发送 READY...")

            while True:
                msg = self.reader.recv_line()
                self.log(f"[LabVIEWTCP] 收到 LabVIEW 消息: {msg[:200]}")

                if msg == "READY":
                    self.is_ready = True
                    self.log("[LabVIEWTCP] 收到 READY，可以开始采集")
                    return {
                        "ok": True,
                        "reason": "ready",
                    }

                if msg == "STOP":
                    return {
                        "ok": False,
                        "reason": "labview_stop",
                    }

                if msg.startswith("ERROR"):
                    return {
                        "ok": False,
                        "reason": msg,
                    }

        except Exception as e:
            self.is_ready = False
            self.log(f"[LabVIEWTCP] 等待 READY 失败: {e}")
            return {
                "ok": False,
                "reason": str(e),
            }

    # ============================================================
    # 二进制采集
    # ============================================================

    def request_measure(
        self,
        command: str = "MEASURE",
        index: Optional[int] = None,
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        """
        发送一次 MEASURE，然后接收 LabVIEW 返回的二进制数据：

            header = 4 字节 U32
            payload = header 指定长度的 DBL 数组二进制数据
        """
        try:
            self._ensure_connected()

            if index is None:
                self.measure_count += 1
                index = self.measure_count
            else:
                self.measure_count = max(self.measure_count, int(index))

            # ----------------------------------------------------
            # 1. 发送 MEASURE 命令
            # ----------------------------------------------------
            if self.send_newline:
                cmd_bytes = (command.strip() + "\n").encode("utf-8")
            else:
                cmd_bytes = command.strip().encode("utf-8")

            self.conn.sendall(cmd_bytes)
            self.log(f"[LabVIEWTCP] 第 {index} 次已发送命令: {cmd_bytes!r}")

            # ----------------------------------------------------
            # 2. 读取 4 字节 header
            # ----------------------------------------------------
            header = self.reader.recv_exactly(4)
            payload_len, header_order = self._parse_payload_len(header)

            self.log(
                f"[LabVIEWTCP] 第 {index} 次 header={header.hex(' ')}, "
                f"payload_len={payload_len}, header_order={header_order}"
            )

            # ----------------------------------------------------
            # 3. 检查 payload_len 是否合理
            # ----------------------------------------------------
            if payload_len <= 0:
                return self._error_result(
                    index=index,
                    reason=f"invalid_payload_len: {payload_len}",
                    payload_len=payload_len,
                    header=header,
                )

            # 防止协议错位后误读一个极大的长度，导致程序卡死或内存暴涨
            if payload_len > 100 * 1024 * 1024:
                return self._error_result(
                    index=index,
                    reason=f"payload_len_too_large: {payload_len}",
                    payload_len=payload_len,
                    header=header,
                )

            if payload_len != self.expected_payload_len:
                self.log(
                    "[LabVIEWTCP] 警告：payload_len 与预期不一致。"
                    f"实际={payload_len}，预期={self.expected_payload_len}。"
                    "如果实际是 8196，通常说明第一个“平化至字符串”前置了数组长度。"
                )

            if payload_len % 8 != 0:
                return self._error_result(
                    index=index,
                    reason=f"payload_len_not_multiple_of_8: {payload_len}",
                    payload_len=payload_len,
                    header=header,
                )

            # ----------------------------------------------------
            # 4. 精确读取 payload
            # ----------------------------------------------------
            payload = self.reader.recv_exactly(payload_len)

            # ----------------------------------------------------
            # 5. 将 payload 解析为 DBL 数组
            # ----------------------------------------------------
            dtype = self._payload_dtype()
            values_np = np.frombuffer(payload, dtype=dtype)
            values = values_np.astype(float).tolist()

            point_count_ok = len(values) == self.expected_points

            # ----------------------------------------------------
            # 6. 保存 CSV
            # ----------------------------------------------------
            csv_path = None
            if save_csv:
                csv_path = save_values_to_csv(
                    values=values,
                    index=index,
                    output_dir=self.output_dir,
                )

            if point_count_ok:
                self.log(
                    f"[LabVIEWTCP] 第 {index} 次采集完成："
                    f"num_points={len(values)}, payload_len={payload_len}"
                )
            else:
                self.log(
                    f"[LabVIEWTCP] 第 {index} 次采集点数异常："
                    f"num_points={len(values)}，expected_points={self.expected_points}"
                )

            return {
                "ok": point_count_ok,
                "reason": "ok" if point_count_ok else "point_count_mismatch",
                "index": index,
                "csv_path": csv_path,
                "values": values,
                "num_points": len(values),
                "payload_len": payload_len,
                "expected_payload_len": self.expected_payload_len,
                "header_raw_hex": header.hex(" "),
                "header_order": header_order,
                "payload_byte_order": self.payload_byte_order,
            }

        except socket.timeout:
            self.log(f"[LabVIEWTCP] 第 {index} 次采集超时")
            return {
                "ok": False,
                "reason": "socket_timeout",
                "index": index,
                "csv_path": None,
                "values": [],
                "num_points": 0,
            }

        except Exception as e:
            self.log(f"[LabVIEWTCP] 第 {index} 次二进制采集失败: {e}")
            return {
                "ok": False,
                "reason": str(e),
                "index": index,
                "csv_path": None,
                "values": [],
                "num_points": 0,
            }

    # ============================================================
    # header / payload 解析
    # ============================================================

    def _parse_payload_len(self, header: bytes) -> tuple[int, str]:
        """
        解析 4 字节 header。

        LabVIEW 第二个“平化至字符串”如果是大端：
            8192 -> 00 00 20 00

        如果是小端：
            8192 -> 00 20 00 00

        这里默认自动判断哪个更接近期望长度。
        """
        if len(header) != 4:
            raise ValueError(f"header 长度错误：{len(header)}")

        big_value = struct.unpack(">I", header)[0]
        little_value = struct.unpack("<I", header)[0]

        if not self.auto_detect_header_byte_order:
            return big_value, "big"

        expected = self.expected_payload_len

        # 优先选等于期望值的解释
        if big_value == expected:
            return big_value, "big"

        if little_value == expected:
            return little_value, "little"

        # 如果都不等于期望值，选更合理的那个
        candidates = []

        if 0 < big_value < 100 * 1024 * 1024:
            candidates.append((abs(big_value - expected), big_value, "big"))

        if 0 < little_value < 100 * 1024 * 1024:
            candidates.append((abs(little_value - expected), little_value, "little"))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, value, order = candidates[0]
            return value, order

        # 都不合理时返回大端解释，后面会报错
        return big_value, "big"

    def _payload_dtype(self) -> str:
        """
        解析 LabVIEW 第一个“平化至字符串”输出的 DBL 数组。
        """
        if self.payload_byte_order == "big":
            return ">f8"

        if self.payload_byte_order == "little":
            return "<f8"

        raise ValueError("payload_byte_order 只能是 'big' 或 'little'")

    # ============================================================
    # 错误结果
    # ============================================================

    def _error_result(
        self,
        index: int,
        reason: str,
        payload_len: Optional[int] = None,
        header: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        self.log(f"[LabVIEWTCP] 第 {index} 次失败：{reason}")

        return {
            "ok": False,
            "reason": reason,
            "index": index,
            "csv_path": None,
            "values": [],
            "num_points": 0,
            "payload_len": payload_len,
            "expected_payload_len": self.expected_payload_len,
            "header_raw_hex": header.hex(" ") if header is not None else None,
        }

    # ============================================================
    # 关闭
    # ============================================================

    def close(self) -> None:
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None

            if self.server_socket is not None:
                try:
                    self.server_socket.close()
                except Exception:
                    pass
                self.server_socket = None

            self.reader = None
            self.addr = None
            self.is_connected = False
            self.is_ready = False
            self.is_server_running = False

        self.log("[LabVIEWTCP] TCP 连接已关闭")

    def _ensure_connected(self) -> None:
        if self.conn is None or self.reader is None or not self.is_connected:
            raise RuntimeError("LabVIEW 尚未连接 TCP Server")


def main():
    """
    单独测试入口。

    使用顺序：
        1. 先运行这个 Python 文件；
        2. 再运行 LabVIEW，让 LabVIEW 连接 Python；
        3. LabVIEW 发送 READY\\n；
        4. Python 控制台输入 123456；
        5. Python 发送 MEASURE；
        6. LabVIEW 返回 4字节header + payload；
        7. Python 保存 CSV。
    """

    server = LabVIEWTCPServer(
        host="127.0.0.1",
        port=65432,
        output_dir="labview_csv_output",
        socket_timeout_s=30.0,

        # 你的光谱如果不是 1024 点，这里改成实际点数
        expected_points=1024,

        # 根据你截图，LabVIEW 左下角像是直接比较 MEASURE，
        # 所以这里先保持 False，不发送换行。
        send_newline=False,

        # 如果 LabVIEW 第一个“平化至字符串”的字节顺序设为“大端/网络字节顺序”，这里保持 big。
        # 如果收到的光谱数值明显异常，再改成 little 试一次。
        payload_byte_order="big",

        # header 自动判断大小端
        auto_detect_header_byte_order=True,
    )

    server.start_server_blocking()

    ready_result = server.wait_for_ready()

    if not ready_result.get("ok", False):
        print("[Test] READY 失败:", ready_result)
        server.close()
        return

    try:
        while True:
            user_input = input("\n请输入 123456 采集；输入 q 退出：").strip()

            if user_input.lower() == "q":
                break

            if user_input != "123456":
                print("[Test] 输入不是 123456，不发送 MEASURE")
                continue

            result = server.request_measure(
                command="MEASURE",
                save_csv=True,
            )

            print("[Test] ok:", result.get("ok"))
            print("[Test] reason:", result.get("reason"))
            print("[Test] num_points:", result.get("num_points"))
            print("[Test] payload_len:", result.get("payload_len"))
            print("[Test] expected_payload_len:", result.get("expected_payload_len"))
            print("[Test] header_raw_hex:", result.get("header_raw_hex"))
            print("[Test] header_order:", result.get("header_order"))
            print("[Test] payload_byte_order:", result.get("payload_byte_order"))
            print("[Test] csv_path:", result.get("csv_path"))

    finally:
        server.close()


if __name__ == "__main__":
    main()