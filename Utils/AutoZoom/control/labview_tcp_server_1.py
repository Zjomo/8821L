# control/labview_tcp_server.py
# 或 logic/labview_tcp_server.py
from __future__ import annotations

import socket
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple


LogCallback = Optional[Callable[[str], None]]


class MessageReader:
    """
    TCP 消息读取器。

    作用：
        从 TCP 字节流中持续读取数据，直到遇到消息结束符。

    支持两种结束符：
        1. 真正换行符：b"\\n"
        2. 字符串形式的反斜杠+n：b"\\\\n"

    注意：
        TCP 是字节流，不保证一次 recv 对应 LabVIEW 一次 TCP Write。
        因此 Python 端必须自己定义消息边界。
    """

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.buffer = b""

    def recv_one_message(self) -> str:
        """
        读取一条以 \\n 或 \\n 字符串结尾的消息。

        返回：
            去掉首尾空白后的字符串。
        """
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
    """
    将 LabVIEW 返回的 DATA 内容解析成数值列表。

    支持格式：
        1. 逗号分隔：
           1107.000000,1106.000000,1106.000000

        2. 分号分隔：
           1107.000000,1106.000000;1108.000000,1109.000000

        3. 混合空格、Tab、回车：
           1107.000000 1106.000000

        4. 带 DATA, 前缀：
           DATA,1107.000000,1106.000000
    """
    if data_text is None:
        return []

    text = str(data_text).strip()

    # 兼容直接传入 DATA,xxx 的情况
    if text.startswith("DATA,"):
        text = text.split(",", 1)[1]

    cleaned = (
        text
        .replace("\r", ",")
        .replace("\n", ",")
        .replace("\\n", ",")
        .replace("\t", ",")
        .replace(" ", ",")
        .replace(";", ",")
    )

    values: List[float] = []

    for item in cleaned.split(","):
        token = item.strip()
        if not token:
            continue

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

    with filepath.open(mode="w", newline="", encoding="utf-8-sig") as f:
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
        4. 向 LabVIEW 发送 MEASURE\\n；
        5. 等待 LabVIEW 返回光谱数据；
        6. 解析数据并保存 CSV。

    支持 LabVIEW 返回两种格式：

        旧格式：
            DATA,1101,1102,1103,...\\n

        新格式：
            DATA_BEGIN\\n
            DATA,1101,1102,1103,...\\n
            DATA_END\\n

    推荐使用新格式。
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
                name="LabVIEWTCP-AcceptThread",
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
                    self.log("[LabVIEWTCP] 收到 STOP")
                    return {
                        "ok": False,
                        "reason": "labview_stop",
                    }

                if msg.startswith("ERROR"):
                    self.log(f"[LabVIEWTCP] 收到 ERROR: {msg}")
                    return {
                        "ok": False,
                        "reason": msg,
                    }

                self.log(f"[LabVIEWTCP] 等待 READY 时收到非 READY 消息，已忽略: {msg[:200]}")

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
        发送一次 MEASURE，并等待 LabVIEW 返回数据。

        支持旧格式：
            DATA,1101,1102,1103,...\\n

        支持新格式：
            DATA_BEGIN\\n
            DATA,1101,1102,1103,...\\n
            DATA_END\\n

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

            # 关键修改：
            # 建议发送 MEASURE\n，而不是只发送 MEASURE。
            # 这样 LabVIEW 端 TCP Read 更容易按一条命令读取。
            command_text = str(command).strip()
            command_to_send = command_text + "\n"

            self.conn.sendall(command_to_send.encode("utf-8"))

            self.log(
                f"[LabVIEWTCP] 第 {index} 次已发送 {command_text}，"
                f"实际发送内容={command_to_send!r}，等待 DATA..."
            )

            while True:
                msg = self.reader.recv_one_message()
                self.log(f"[LabVIEWTCP] 收到 LabVIEW 消息前200字符: {msg[:200]}")

                # ------------------------------------------------------------
                # 旧格式：DATA,xxx,xxx,xxx
                # ------------------------------------------------------------
                if msg.startswith("DATA,"):
                    return self._handle_single_line_data(
                        msg=msg,
                        index=index,
                        save_csv=save_csv,
                    )

                # ------------------------------------------------------------
                # 新格式：
                # DATA_BEGIN
                # DATA,xxx,xxx,xxx
                # DATA_END
                # ------------------------------------------------------------
                if msg == "DATA_BEGIN":
                    return self._handle_data_block(
                        index=index,
                        save_csv=save_csv,
                    )

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

                if msg.startswith("ERROR"):
                    self.log(f"[LabVIEWTCP] LabVIEW 返回错误: {msg}")
                    return {
                        "ok": False,
                        "reason": msg,
                        "index": index,
                        "csv_path": None,
                        "values": [],
                        "num_points": 0,
                    }

                # 这里仍然保留日志，但不会中断流程
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

    def _handle_single_line_data(
        self,
        msg: str,
        index: int,
        save_csv: bool,
    ) -> Dict[str, Any]:
        """
        处理旧格式：
            DATA,1101,1102,1103,...
        """
        data_text = msg.split(",", 1)[1]
        values = parse_labview_data(data_text)

        csv_path = None
        if save_csv:
            csv_path, values = save_data_to_csv(
                data_text=data_text,
                index=index,
                output_dir=self.output_dir,
            )

        self.log(f"[LabVIEWTCP] 第 {index} 次收到旧格式 DATA")
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

    def _handle_data_block(
        self,
        index: int,
        save_csv: bool,
    ) -> Dict[str, Any]:
        """
        处理新格式：
            DATA_BEGIN
            DATA,1101,1102,1103,...
            DATA_END

        同时兼容 LabVIEW 把 DATA 数组字符串中间换行的情况：

            DATA_BEGIN
            DATA,1101,1102,1103
            1104,1105,1106
            1107,1108
            DATA_END

        DATA_BEGIN 和 DATA_END 之间的所有非 ERROR 行都会合并后解析。
        """
        data_lines: List[str] = []

        while True:
            line = self.reader.recv_one_message()
            self.log(f"[LabVIEWTCP] DATA块行前200字符: {line[:200]}")

            if line == "DATA_END":
                break

            if line == "STOP":
                self.log("[LabVIEWTCP] DATA块接收过程中收到 STOP")
                return {
                    "ok": False,
                    "reason": "labview_stop",
                    "index": index,
                    "csv_path": None,
                    "values": [],
                    "num_points": 0,
                }

            if line.startswith("ERROR"):
                self.log(f"[LabVIEWTCP] DATA块接收过程中收到 ERROR: {line}")
                return {
                    "ok": False,
                    "reason": line,
                    "index": index,
                    "csv_path": None,
                    "values": [],
                    "num_points": 0,
                }

            data_lines.append(line)

        raw_text_parts: List[str] = []

        for line in data_lines:
            if line.startswith("DATA,"):
                raw_text_parts.append(line.split(",", 1)[1])
            else:
                # 兼容数组字符串中间换行后，没有 DATA, 前缀的后续行
                raw_text_parts.append(line)

        data_text = ",".join(raw_text_parts)
        values = parse_labview_data(data_text)

        csv_path = None
        if save_csv:
            csv_path, values = save_data_to_csv(
                data_text=data_text,
                index=index,
                output_dir=self.output_dir,
            )

        self.log(f"[LabVIEWTCP] 第 {index} 次收到 DATA_BEGIN/DATA_END 数据块")
        self.log(f"[LabVIEWTCP] DATA块原始行数: {len(data_lines)}")
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
            "data_block_lines": data_lines,
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
        python control/labview_tcp_server.py

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

            result = server.request_measure(
                command="MEASURE",
                save_csv=True,
            )

            print("[Test] result ok:", result.get("ok"))
            print("[Test] reason:", result.get("reason"))
            print("[Test] csv_path:", result.get("csv_path"))
            print("[Test] num_points:", result.get("num_points"))

    finally:
        server.close()


if __name__ == "__main__":
    main()