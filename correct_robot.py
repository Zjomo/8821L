import argparse
import importlib
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np


warnings.warn(
    "correct_robot.py is DEPRECATED. All functionality has been superseded by SpotZoom.py. "
    "Use SpotZoom.py with --xy-driver dryrun for dry-run mode, or --xy-driver newport/thorlabs "
    "for hardware control. This file will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


LOGGER = logging.getLogger("RealTimeSpotController")


def _resolve_model_path(explicit_model: Optional[str]) -> str:
    candidates = []
    if explicit_model:
        candidates.append(Path(explicit_model))
    env_model = os.environ.get("SPOTZOOM_MODEL_PATH")
    if env_model:
        candidates.append(Path(env_model))

    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir / "best.pt",
            script_dir / "best_all.pt",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "YOLO model not found. Set SPOTZOOM_MODEL_PATH or pass --model-path."
    )


def _import_ultralytics_yolo():
    try:
        module = importlib.import_module("ultralytics")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install with: python -m pip install ultralytics"
        ) from exc
    return module.YOLO


def _import_thorlabs_class():
    try:
        devices = importlib.import_module("pylablib.devices")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: pylablib. Install with: python -m pip install pylablib"
        ) from exc
    return devices.Thorlabs.KinesisPiezoMotor


def _import_pyautogui():
    try:
        return importlib.import_module("pyautogui")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: pyautogui. Install with: python -m pip install pyautogui"
        ) from exc


class DryRunThorlabsDevice:
    """Mock Thorlabs device that logs calls instead of moving hardware.

    Used when ``dry_run=True`` to allow testing the control loop without
    a physical KinesisPiezoMotor connected.
    """
    def open(self) -> None:
        LOGGER.info("[dryrun] controller opened")

    def move_by(self, distance: int, auto_enable: bool, channel: int) -> None:
        LOGGER.info(
            "[dryrun] move channel=%s distance=%s auto_enable=%s",
            channel,
            distance,
            auto_enable,
        )

    def stop(self, channel: int) -> None:
        LOGGER.info("[dryrun] stop channel=%s", channel)

    def close(self) -> None:
        LOGGER.info("[dryrun] controller closed")


class RealTimeSpotController:
    """Real-time spot position controller using YOLO detection and Thorlabs piezo motor.

    Captures screen regions, detects a spot via YOLO, and drives a Thorlabs
    KinesisPiezoMotor to move the spot toward a target position.
    """

    CHANNEL_MAP = {"vertical": 3, "horizontal": 4}

    def __init__(
        self,
        device_id: str = "97101208",
        model_path: Optional[str] = None,
        dry_run: bool = False,
        preview: bool = True,
    ):
        """Initialize controller with device ID, model path, and run options.

        Args:
            device_id: Serial number of the Thorlabs KinesisPiezoMotor device.
            model_path: Explicit path to the YOLO model file. If *None*, the
                model is resolved via ``_resolve_model_path``.
            dry_run: If *True*, use a mock device instead of real hardware.
            preview: If *True*, show an OpenCV preview window during detection.
        """
        self.device_id = device_id
        self.dry_run = dry_run
        self.preview = preview

        self.pyautogui = _import_pyautogui()
        self.thorlabs_device = self.connect_controller()
        self.model_path = _resolve_model_path(model_path)
        self.yolo = self.init_yolo_model()

        self.region = (118, 93, 1112, 888)
        self.target_pos = (556, 444)
        self.threshold = 10
        self.step_size = 500
        self.stabilization_time = 0.2

        self.current_pos = {"x": 0, "y": 0}
        self.running = True
        self.lock = threading.Lock()
        self.target_reached = threading.Event()
        self.has_detection = threading.Event()
        self.threads = []

    def connect_controller(self) -> Any:
        """Open a connection to the Thorlabs piezo motor controller.

        In dry-run mode a mock device is returned instead.  Note that
        ``pylablib.devices.Thorlabs.KinesisPiezoMotor.open()`` does not
        accept a timeout parameter; if the device is unresponsive the call
        may block indefinitely.  Ensure the USB connection is healthy before
        invoking this method.

        Returns:
            An open Thorlabs device instance (or a ``DryRunThorlabsDevice``).
        """
        if self.dry_run:
            device = DryRunThorlabsDevice()
            device.open()
            return device

        motor_cls = _import_thorlabs_class()
        try:
            device = motor_cls(self.device_id)
            device.open()
            LOGGER.info("Thorlabs controller connected: %s", self.device_id)
            return device
        except Exception as exc:
            raise RuntimeError(f"Thorlabs controller connection failed: {exc}") from exc

    def init_yolo_model(self) -> Any:
        yolo_cls = _import_ultralytics_yolo()
        model = yolo_cls(self.model_path)
        LOGGER.info("YOLO model loaded: %s", self.model_path)
        return model

    def move_axis(self, axis: str, direction: str) -> None:
        """Move the specified motor axis by one step in the given direction.

        Args:
            axis: Either ``"vertical"`` or ``"horizontal"``.
            direction: One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.

        Raises:
            ValueError: If *axis* is not recognised.
            RuntimeError: If the motor move command fails (e.g. USB disconnect).
        """
        if axis not in self.CHANNEL_MAP:
            raise ValueError(f"Unknown axis: {axis}")
        sign = 1 if direction in ("up", "left") else -1
        try:
            self.thorlabs_device.move_by(
                distance=sign * self.step_size,
                auto_enable=True,
                channel=self.CHANNEL_MAP[axis],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Motor move failed on {axis} axis. "
                f"Device may be disconnected — check USB connection. Original error: {exc}"
            ) from exc

    def stop_all(self) -> None:
        """Stop all motor channels and signal the control loop to exit."""
        with self.lock:
            if self.thorlabs_device:
                for channel in (3, 4):
                    try:
                        self.thorlabs_device.stop(channel=channel)
                    except Exception as exc:
                        LOGGER.warning("Stop channel %d failed: %s", channel, exc)
            self.running = False
            self.target_reached.set()

    def stop_axis(self, axis: str) -> None:
        """Stop a single motor axis.

        Args:
            axis: Either ``"vertical"`` or ``"horizontal"``.
        """
        channel = self.CHANNEL_MAP[axis]
        try:
            self.thorlabs_device.stop(channel=channel)
        except Exception as exc:
            LOGGER.warning("Stop axis failed axis=%s channel=%s error=%s", axis, channel, exc)

    def _extract_spot(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Run YOLO inference on *frame* and return the centre of the first detected spot.

        Returns:
            ``(x, y)`` pixel coordinates of the spot centre, or *None* if no
            spot of class 0 is detected.
        """
        results = self.yolo(frame, verbose=False)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for *xyxy, _conf, cls in boxes.data.cpu().numpy():
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    return (x1 + x2) // 2, (y1 + y2) // 2
        return None

    def detection_worker(self) -> None:
        """Continuously capture screenshots and detect the spot position.

        Runs in a dedicated thread.  Updates ``self.current_pos`` and sets
        ``self.has_detection`` when a spot is found.  If the spot reaches the
        target position, ``stop_all`` is called.  Temporary detection errors
        are logged and retried; the thread only exits on fatal conditions.
        """
        while self.running:
            try:
                screenshot = self.pyautogui.screenshot(region=self.region)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                spot_pos = self._extract_spot(frame)

                reached = False
                with self.lock:
                    if spot_pos is not None:
                        self.current_pos["x"] = spot_pos[0]
                        self.current_pos["y"] = spot_pos[1]
                        self.has_detection.set()
                        dx = abs(self.target_pos[0] - spot_pos[0])
                        dy = abs(self.target_pos[1] - spot_pos[1])
                        if dx <= self.threshold and dy <= self.threshold:
                            reached = True

                if reached:
                    LOGGER.info("Target reached at %s", spot_pos)
                    self.stop_all()
                    continue

                if self.preview:
                    if spot_pos is not None:
                        cv2.circle(frame, spot_pos, 5, (0, 255, 0), -1)
                    cv2.imshow("Control Panel", frame)
                    cv2.moveWindow("Control Panel", 1300, 100)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_all()
                        break
            except Exception as exc:
                LOGGER.warning("Detection error (will retry): %s", exc)
                time.sleep(0.1)
                continue

    def control_worker(self) -> None:
        """Drive the motor axes to bring the spot toward the target position.

        Runs in a dedicated thread.  Waits for at least one detection before
        issuing move commands.  Exits when ``self.running`` is cleared or
        ``self.target_reached`` is set.
        """
        while self.running:
            try:
                if not self.has_detection.wait(timeout=0.5):
                    continue

                with self.lock:
                    current_x = self.current_pos["x"]
                    current_y = self.current_pos["y"]

                dx = self.target_pos[0] - current_x
                dy = self.target_pos[1] - current_y

                if abs(dy) <= self.threshold:
                    self.stop_axis("vertical")
                else:
                    direction = "down" if dy > 0 else "up"
                    self.move_axis("vertical", direction)

                if abs(dx) <= self.threshold:
                    self.stop_axis("horizontal")
                else:
                    direction = "right" if dx > 0 else "left"
                    self.move_axis("horizontal", direction)

                if self.target_reached.is_set():
                    break
                time.sleep(self.stabilization_time)
            except Exception as exc:
                LOGGER.exception("Control thread failed: %s", exc)
                self.stop_all()
                break

    def start(self, max_runtime_s: Optional[float] = None) -> None:
        """Start the detection and control worker threads.

        Blocks the calling thread until the control loop finishes or
        ``max_runtime_s`` elapses.  Handles ``KeyboardInterrupt`` gracefully.

        Args:
            max_runtime_s: If provided, the loop will self-stop after this
                many seconds.
        """
        detection_thread = threading.Thread(target=self.detection_worker, name="detection_worker")
        control_thread = threading.Thread(target=self.control_worker, name="control_worker")
        self.threads = [detection_thread, control_thread]

        for thread in self.threads:
            thread.start()

        start_time = time.time()
        try:
            while self.running:
                if max_runtime_s is not None and (time.time() - start_time) >= max_runtime_s:
                    LOGGER.info("Max runtime reached (%.2fs), stopping", max_runtime_s)
                    self.stop_all()
                    break

                with self.lock:
                    LOGGER.info(
                        "Current position: (%s, %s)",
                        self.current_pos["x"],
                        self.current_pos["y"],
                    )
                time.sleep(0.5)
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted by user")
            self.stop_all()

    def shutdown(self) -> None:
        """Gracefully shut down the controller.

        Stops all motors, joins worker threads, closes the OpenCV preview
        window, and closes the Thorlabs device connection.
        """
        self.stop_all()
        for thread in self.threads:
            thread.join(timeout=2)
        cv2.destroyAllWindows()
        if self.thorlabs_device:
            try:
                self.thorlabs_device.close()
            except Exception as exc:
                LOGGER.warning("Controller close failed: %s", exc)


def check_environment(model_path: Optional[str], dry_run: bool = False) -> int:
    """Verify that all required dependencies and the model file are available.

    Args:
        model_path: Explicit model path to resolve (may be *None*).
        dry_run: If *True*, the Thorlabs dependency check is skipped.

    Returns:
        0 if everything is OK, 1 if one or more checks failed.
    """
    errors = []
    try:
        _import_ultralytics_yolo()
    except Exception as exc:
        errors.append(str(exc))

    try:
        _import_pyautogui()
    except Exception as exc:
        errors.append(str(exc))

    if not dry_run:
        try:
            _import_thorlabs_class()
        except Exception as exc:
            errors.append(str(exc))

    try:
        resolved_model = _resolve_model_path(model_path)
        LOGGER.info("Model resolved: %s", resolved_model)
    except Exception as exc:
        errors.append(str(exc))

    if errors:
        for item in errors:
            LOGGER.error(item)
        return 1
    LOGGER.info("Environment check passed")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the spot controller."""
    parser = argparse.ArgumentParser(description="Real-time spot control using YOLO + Thorlabs")
    parser.add_argument("--device-id", default="97101208", help="Thorlabs device id")
    parser.add_argument("--model-path", default=None, help="Path to YOLO model")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run device")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--max-runtime", type=float, default=None, help="Optional max runtime in seconds")
    parser.add_argument("--check-env", action="store_true", help="Only check dependencies and model path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: parse arguments, create the controller, and run.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.check_env:
        return check_environment(args.model_path, dry_run=args.dry_run)

    controller = None
    try:
        controller = RealTimeSpotController(
            device_id=args.device_id,
            model_path=args.model_path,
            dry_run=args.dry_run,
            preview=not args.no_preview,
        )
        controller.start(max_runtime_s=args.max_runtime)
        return 0
    except Exception as exc:
        LOGGER.exception("Controller failed: %s", exc)
        return 1
    finally:
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
