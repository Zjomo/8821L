# vision/capture.py
import time
import logging
import numpy as np
import pyautogui
import queue

logger = logging.getLogger(__name__)

class ScreenCaptureWorker:
    def __init__(self, image_queue, stop_event, is_detecting_getter, capture_area):
        self.image_queue = image_queue
        self.stop_event = stop_event
        self.is_detecting_getter = is_detecting_getter
        self.capture_area = capture_area  # (left, top, width, height)

    def run(self):
        logger.info("图像采集线程启动")
        while not self.stop_event.is_set() and self.is_detecting_getter():
            try:
                frame = pyautogui.screenshot(region=self.capture_area)
                image = np.array(frame)

                # 队列满则丢弃旧帧
                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.image_queue.put(image)
                time.sleep(0.03)
            except Exception as e:
                logger.error(f"采集错误: {e}")
                time.sleep(0.1)
        logger.info("图像采集线程退出")
