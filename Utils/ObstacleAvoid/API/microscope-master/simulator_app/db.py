#!/usr/bin/env python3

"""SQLite 持久化层。

保存模拟显微镜的运行状态：
* ``device_state``      —— 各轴当前位置、相机曝光/增益等键值状态（重启后恢复）
* ``saved_positions``   —— 用户保存的命名位置（预设点位）
* ``move_log``          —— 位移台移动历史
* ``acquisitions``      —— 相机快照（PNG BLOB）及采集时的设备状态
"""

import io
import os
import sqlite3
import threading
from typing import Any, List, Optional, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS device_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_positions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    x          REAL NOT NULL,
    y          REAL NOT NULL,
    z          REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS move_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    source   TEXT NOT NULL,
    x        REAL NOT NULL,
    y        REAL NOT NULL,
    z        REAL NOT NULL,
    note     TEXT
);
CREATE TABLE IF NOT EXISTS acquisitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    exposure_ms REAL NOT NULL,
    gain        REAL NOT NULL,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    z           REAL NOT NULL,
    image       BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_objects (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    label    TEXT NOT NULL,
    shape    TEXT NOT NULL,
    cx       REAL NOT NULL,
    cy       REAL NOT NULL,
    size     REAL NOT NULL,
    angle    REAL NOT NULL,
    params   TEXT
);
"""


class Database:
    """线程安全的 SQLite 封装（GUI 线程与采集线程共用同一连接）。"""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(os.fspath(path)))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            # SQLite's user_version is a lightweight migration marker and
            # keeps future schema changes explicit without breaking existing
            # simulator databases.
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                self._conn.execute("PRAGMA user_version = 1")
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "Database":
        if self._closed:
            raise RuntimeError("database is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ------------------------------------------------------------------
    # device_state：键值状态
    # ------------------------------------------------------------------
    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM device_state WHERE key = ?", (key,)
            ).fetchone()
        return default if row is None else row[0]

    def get_state_float(self, key: str, default: float) -> float:
        value = self.get_state(key)
        return default if value is None else float(value)

    def set_state(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO device_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    # ------------------------------------------------------------------
    # saved_positions：命名位置
    # ------------------------------------------------------------------
    def add_position(self, name: str, x: float, y: float, z: float) -> bool:
        """返回 False 表示同名位置已存在。"""
        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM saved_positions WHERE name = ?", (name,)
            ).fetchone()
            if exists:
                return False
            self._conn.execute(
                "INSERT INTO saved_positions (name, x, y, z) VALUES (?, ?, ?, ?)",
                (name, x, y, z),
            )
        return True

    def list_positions(self) -> Sequence[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, name, x, y, z, created_at FROM saved_positions ORDER BY id"
            ).fetchall()

    def get_position(self, pos_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, name, x, y, z FROM saved_positions WHERE id = ?", (pos_id,)
            ).fetchone()

    def delete_position(self, pos_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM saved_positions WHERE id = ?", (pos_id,))

    # ------------------------------------------------------------------
    # move_log：移动历史
    # ------------------------------------------------------------------
    def log_move(self, source: str, x: float, y: float, z: float, note: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO move_log (source, x, y, z, note) VALUES (?, ?, ?, ?, ?)",
                (source, x, y, z, note),
            )

    def recent_moves(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, source, x, y, z, note FROM move_log "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return list(reversed(rows))

    # ------------------------------------------------------------------
    # acquisitions：快照
    # ------------------------------------------------------------------
    def save_acquisition(
        self, png_bytes: bytes, exposure_ms: float, gain: float, x: float, y: float, z: float
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO acquisitions (exposure_ms, gain, x, y, z, image) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exposure_ms, gain, x, y, z, png_bytes),
            )
            return cur.lastrowid

    def encode_png(self, array: "Any") -> bytes:
        """把 numpy 图像编码为 PNG（供 save_acquisition 使用）。"""
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # sample_objects：样本图层对象（掩码/衬底/障碍物）
    # ------------------------------------------------------------------
    def replace_sample_objects(self, rows) -> None:
        """整体替换样本对象清单。rows 为 dict 列表。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sample_objects")
            self._conn.executemany(
                "INSERT INTO sample_objects (category, label, shape, cx, cy, size, angle, params) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r["category"], r["label"], r["shape"],
                        float(r["cx"]), float(r["cy"]),
                        float(r["size"]), float(r["angle"]),
                        r.get("params", ""),
                    )
                    for r in rows
                ],
            )

    def list_sample_objects(self) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, category, label, shape, cx, cy, size, angle, params "
                "FROM sample_objects ORDER BY id"
            ).fetchall()
