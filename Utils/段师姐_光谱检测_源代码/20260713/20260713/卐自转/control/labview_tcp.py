import socket


HOST = "127.0.0.1"
PORT = 65432


class LineReader:
    def __init__(self, conn):
        self.conn = conn
        self.buffer = b""

    def recv_line(self):
        while True:
            # 同时识别：
            # 1. 真正换行：b"\n"
            # 2. LabVIEW 字符串里的字面量：b"\\n"
            candidates = []

            pos_real = self.buffer.find(b"\n")
            if pos_real != -1:
                candidates.append((pos_real, b"\n"))

            pos_literal = self.buffer.find(b"\\n")
            if pos_literal != -1:
                candidates.append((pos_literal, b"\\n"))

            if candidates:
                pos, sep = min(candidates, key=lambda x: x[0])
                line = self.buffer[:pos]
                self.buffer = self.buffer[pos + len(sep):]
                return line.decode("utf-8", errors="ignore").strip()

            chunk = self.conn.recv(4096)
            if not chunk:
                raise ConnectionError("LabVIEW 已断开连接")

            print(f"[TCP] 原始字节长度: {len(chunk)}")
            self.buffer += chunk


def wait_for_ready(reader):
    print("[TCP] 等待 LabVIEW READY...")

    while True:
        line = reader.recv_line()
        print(f"[TCP] 收到: {line[:120]}")

        if line.upper() == "READY":
            print("[TCP] LabVIEW READY，可以开始采集")
            return

        if line.upper().startswith("ERROR"):
            raise RuntimeError(f"LabVIEW 返回错误: {line}")

        if line == "":
            continue

        print(f"[TCP] READY 前收到其他消息，已忽略: {line[:120]}")


def parse_array_line(line, prefix):
    """
    解析：
        RAW_Y,100,120,130,125
        FIT_Y,101.2,119.5,130.1,124.8

    兼容 RAW_Y / RAW_y / raw_y。
    """
    line = line.strip()

    if "," not in line:
        raise ValueError(f"不是数组行: {line[:120]}")

    head, rest = line.split(",", 1)

    if head.upper() != prefix.upper():
        raise ValueError(f"期望 {prefix}, 但收到: {line[:120]}")

    values = []
    for x in rest.split(","):
        x = x.strip()
        if x == "":
            continue
        values.append(float(x))

    return values


def request_measure(conn, reader, index):
    conn.sendall(b"MEASURE\n")
    print(f"[TCP] 第 {index} 次已发送 MEASURE")

    raw_y = None
    fit_y = None

    while raw_y is None or fit_y is None:
        line = reader.recv_line()
        print(f"[TCP] 收到一行: {line[:120]}")

        if line == "":
            continue

        if line.upper() == "READY":
            print("[TCP] 收到 READY，已忽略")
            continue

        if line.upper().startswith("RAW_Y,"):
            raw_y = parse_array_line(line, "RAW_Y")
            print(f"[TCP] 已解析 RAW_Y，点数={len(raw_y)}")
            continue

        if line.upper().startswith("FIT_Y,"):
            fit_y = parse_array_line(line, "FIT_Y")
            print(f"[TCP] 已解析 FIT_Y，点数={len(fit_y)}")
            continue

        if line.upper().startswith("ERROR"):
            raise RuntimeError(f"LabVIEW 返回错误: {line}")

        print(f"[TCP] 未识别消息，已忽略: {line[:120]}")

    print(f"[TCP] 第 {index} 次采集完成")
    print(f"[TCP] raw_y 点数: {len(raw_y)}")
    print(f"[TCP] fit_y 点数: {len(fit_y)}")

    return {
        "raw_y": raw_y,
        "fit_y": fit_y,
    }


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print(f"[TCP] Python TCP 服务器已启动: {HOST}:{PORT}")
        print("[TCP] 等待 LabVIEW 连接...")

        conn, addr = server.accept()

        with conn:
            print(f"[TCP] LabVIEW 已连接: {addr}")
            reader = LineReader(conn)

            # 关键：先读取并消费 LabVIEW 的 READY
            wait_for_ready(reader)

            index = 1

            while True:
                cmd = input(f"\n输入 123456 开始第 {index} 次采集，输入 q 退出：").strip()

                if cmd.lower() == "q":
                    print("[TCP] 用户退出")
                    break

                if cmd != "123456":
                    print("[TCP] 输入不正确")
                    continue

                result = request_measure(conn, reader, index)

                # 这里可以保存 result["raw_y"] 和 result["fit_y"]

                index += 1


if __name__ == "__main__":
    main()