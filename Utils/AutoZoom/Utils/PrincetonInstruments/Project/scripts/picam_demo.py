"""PICam 直接控制演示脚本。

无需修改现有工作流即可验证 PI 直接控制功能。

运行方式：
    python scripts/picam_demo.py --backend mock
    python scripts/picam_demo.py --backend demo
    python scripts/picam_demo.py --backend picam
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from patches.measurement_workflow_adapter import (
    create_picam_adapter_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="PI 光谱仪直接控制演示")
    parser.add_argument(
        "--backend",
        choices=["picam", "demo", "mock"],
        default="mock",
        help="后端类型",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.1,
        help="曝光时间（秒）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="采集次数",
    )
    parser.add_argument(
        "--dll",
        type=str,
        default=None,
        help="Picam.dll 路径（可选）",
    )
    args = parser.parse_args()

    config = {
        "spectrometer_backend": args.backend,
        "tcp_output_dir": "demo_csv_output",
        "picam_dll_path": args.dll,
        "picam_exposure": args.exposure,
        "picam_temperature": -25.0,
    }

    print(f"使用后端: {args.backend}")
    adapter = create_picam_adapter_from_config(config)
    adapter.start_server_blocking()

    ready = adapter.wait_for_ready()
    print("READY:", ready)
    if not ready["ok"]:
        adapter.close()
        return 1

    try:
        for i in range(args.count):
            result = adapter.request_measure(index=i + 1)
            if result["ok"]:
                print(
                    f"  [{i+1}/{args.count}] 点数={result['num_points']}, "
                    f"raw_peak={result.get('raw_peak')}, "
                    f"fit_peak={result.get('fit_peak')}, "
                    f"csv={result.get('csv_path')}"
                )
            else:
                print(f"  [{i+1}/{args.count}] 失败: {result['reason']}")
    finally:
        adapter.close()

    print("演示结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
