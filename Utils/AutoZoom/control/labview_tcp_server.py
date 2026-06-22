# logic/labview_tcp_server.py
from __future__ import annotations

import socket
import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple


LogCallback = Optional[Callable[[str], None]]


class MessageReader:

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.buffer = b""

    def recv_one_message(self) -> str:
        while True:
            for sep in (b"\n", b"\\n"):
                if sep in self.buffer:
                    msg, self.buffer = self.buffer.split(sep, 1)
                    return msg.decode("utf-8", errors="ignore").strip()

            chunk = self.conn.recv(4096)

            if not chunk:
                raise ConnectionError("LabVIEW 已断开连接")

            self.buffer += chunk


def parse_labview_data(data_text: str) -> List[float]:

    cleaned = (
        data_text
        .replace("\r", "")
        .replace("\n", "")
        .replace("\\n", "")
        .replace("\t", ",")
        .replace(" ", ",")
    )

    tokens = []

    for row in cleaned.split(";"):
        for item in row.split(","):
            item = item.strip()
            if item:
                tokens.append(item)

    values = []

    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            # 无法转换的内容直接跳过
            pass

    return values


def save_data_to_csv(
    data_text: str,
    index: int,
    output_dir: str | Path = "labview_csv_output",
) -> Tuple[str, List[float]]:
    """
    将一次 LabVIEW 返回的数据保存为一个 CSV 文件。

    返回：
        csv_path, values
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"measure_{index:03d}_{timestamp}.csv"
    filepath = output_dir / filename

    values = parse_labview_data(data_text)

    with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["measure_index", index])
        writer.writerow(["timestamp", timestamp])
        writer.writerow(["num_points", len(values)])
        writer.writerow([])
        writer.writerow(["point_index", "value"])

        for i, value in enumerate(values):
            writer.writerow([i, value])

    return str(filepath), values


class LabVIEWTCPServer:
    """
    LabVIEW TCP 通信模块。

    功能：
        1. Python 作为 TCP Server；
        2. 等待 LabVIEW 连接；
        3. 等待 LabVIEW 发送 READY；
        4. 向 LabVIEW 发送 MEASURE；
        5. 等待 LabVIEW 返回 DATA,...；
        6. 解析数据并保存 CSV。

    典型用法：

        server = LabVIEWTCPServer(host="127.0.0.1", port=65432)
        server.start_server_async()

        # 等 LabVIEW 连接后：
        server.wait_for_ready()

        # 请求一次采集：
        result = server.request_measure()
        print(result["csv_path"])
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 65432,
        output_dir: str | Path = "labview_csv_output",
        on_log: LogCallback = None,
    ):
        self.host = host
        self.port = int(port)
        self.output_dir = Path(output_dir)
        self.on_log = on_log

        self.server_socket: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr = None
        self.reader: Optional[MessageReader] = None

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
    # Server 启动 / 等待连接
    # ============================================================

    def start_server_async(self) -> None:
        """
        非阻塞启动 TCP Server。

        GUI 中应该调用这个方法。
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
        阻塞方式启动 TCP Server，并等待 LabVIEW 连接。

        注意：
            这个函数会阻塞，GUI 中不要直接调用。
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

            with self._lock:
                self.conn = conn
                self.addr = addr
                self.reader = MessageReader(conn)
                self.is_connected = True

            self.log(f"[LabVIEWTCP] LabVIEW 已连接: {addr}")

        except OSError as e:
            self.log(f"[LabVIEWTCP] TCP Server 启动失败: {e}")
            self.close()
        except Exception as e:
            self.log(f"[LabVIEWTCP] TCP Server 异常: {e}")
            self.close()

    # ============================================================
    # READY 等待
    # ============================================================

    def wait_for_ready(self) -> Dict[str, Any]:
        """
        等待 LabVIEW 发送 READY。

        返回：
            {
                "ok": True / False,
                "reason": "...",
            }
        """
        try:
            self._ensure_connected()

            self.log("[LabVIEWTCP] 等待 LabVIEW Setup 完成并发送 READY...")

            while True:
                msg = self.reader.recv_one_message()
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

        except Exception as e:
            self.is_ready = False
            self.log(f"[LabVIEWTCP] 等待 READY 失败: {e}")
            return {
                "ok": False,
                "reason": str(e),
            }

    # ============================================================
    # 发送 MEASURE 并接收 DATA
    # ============================================================

    def request_measure(
        self,
        command: str = "MEASURE",
        index: Optional[int] = None,
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        """
        发送一次 MEASURE，并等待 LabVIEW 返回 DATA。

        参数：
            command:
                默认发送 "MEASURE"。

            index:
                当前采集编号。如果不传，自动递增。

            save_csv:
                是否保存 CSV。

        返回：
            {
                "ok": True / False,
                "reason": "...",
                "index": 1,
                "csv_path": "...",
                "values": [...],
                "num_points": ...
            }
        """
        try:
            self._ensure_connected()

            if index is None:
                self.measure_count += 1
                index = self.measure_count
            else:
                self.measure_count = max(self.measure_count, int(index))

            cmd_bytes = command.encode("utf-8")
            self.conn.sendall(cmd_bytes)

            self.log(f"[LabVIEWTCP] 第 {index} 次已发送 {command}，等待 DATA...")

            while True:
                msg = self.reader.recv_one_message()

                self.log(f"[LabVIEWTCP] 收到 LabVIEW 消息前200字符: {msg[:200]}")

                if msg.startswith("DATA,"):
                    data_text = msg.split(",", 1)[1]
                    values = parse_labview_data(data_text)

                    csv_path = None

                    if save_csv:
                        csv_path, values = save_data_to_csv(
                            data_text=data_text,
                            index=index,
                            output_dir=self.output_dir,
                        )

                    self.log(f"[LabVIEWTCP] 第 {index} 次收到 DATA")
                    self.log(f"[LabVIEWTCP] 数据点数量: {len(values)}")

                    if csv_path:
                        self.log(f"[LabVIEWTCP] CSV 已保存: {csv_path}")

                    return {
                        "ok": True,
                        "reason": "ok",
                        "index": index,
                        "csv_path": csv_path,
                        "values": values,
                        "num_points": len(values),
                        "raw_data_text": data_text,
                    }

                if msg == "READY":
                    self.is_ready = True
                    self.log("[LabVIEWTCP] 收到 READY，但当前等待 DATA，已忽略")
                    continue

                if msg == "STOP":
                    self.log("[LabVIEWTCP] LabVIEW 请求 STOP")
                    return {
                        "ok": False,
                        "reason": "labview_stop",
                        "index": index,
                        "csv_path": None,
                        "values": [],
                        "num_points": 0,
                    }

                self.log(f"[LabVIEWTCP] 非 DATA 消息，已忽略: {msg[:200]}")

        except Exception as e:
            self.log(f"[LabVIEWTCP] request_measure 失败: {e}")
            return {
                "ok": False,
                "reason": str(e),
                "index": index,
                "csv_path": None,
                "values": [],
                "num_points": 0,
            }

    # ============================================================
    # 关闭
    # ============================================================

    def close(self) -> None:
        """
        关闭 TCP 连接和 Server。
        """
        with self._lock:
            try:
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
            finally:
                self.reader = None
                self.addr = None
                self.is_connected = False
                self.is_ready = False

            try:
                if self.server_socket is not None:
                    try:
                        self.server_socket.close()
                    except Exception:
                        pass
                    self.server_socket = None
            finally:
                self.is_server_running = False

        self.log("[LabVIEWTCP] TCP 连接已关闭")

    # ============================================================
    # 内部检查
    # ============================================================

    def _ensure_connected(self) -> None:
        if self.conn is None or self.reader is None or not self.is_connected:
            raise RuntimeError("LabVIEW 尚未连接 TCP Server")


def main():
    """
    单独测试用。

    运行：
        python logic/labview_tcp_server.py

    流程：
        1. 启动 Python TCP Server；
        2. LabVIEW 连接；
        3. 等 LabVIEW 发送 READY；
        4. 控制台输入 123456 触发采集。
    """

    server = LabVIEWTCPServer(
        host="127.0.0.1",
        port=65432,
        output_dir="labview_csv_output",
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

            result = server.request_measure()

            print("[Test] result ok:", result.get("ok"))
            print("[Test] csv_path:", result.get("csv_path"))
            print("[Test] num_points:", result.get("num_points"))

    finally:
        server.close()


if __name__ == "__main__":
    main()