import cv2
import numpy as np
import pyautogui
import time
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from pylablib.devices import Newport
import threading
import queue
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rid_model_path = 'model_24.pth'

# 初始化控制器
def init_controller():
    global controller
    controller = Newport.Picomotor8742(backend='auto', multiaddr=False, scan=True)
    usb_devices_num = Newport.get_usb_devices_number_picomotor()
    logging.info(f"Newport Picomotor设备数量: {usb_devices_num}")
    controller.setup_velocity(axis='all', speed=800, accel=800, addr=None)

# 聚焦分数计算函数
def calculate_focus_score(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    focus_score = np.var(laplacian)
    return focus_score

# 判断是否聚焦的函数
def is_focused(image, threshold=35):
    focus_score = calculate_focus_score(image)
    return focus_score > threshold

# 捕获屏幕图像
def capture_screen(capture_area):
    start_time = time.time()
    screenshot = pyautogui.screenshot(region=capture_area)
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    logging.info(f"捕获屏幕时间:{time.time() - start_time:.4f}s")
    return image

# 检查图像是否有效（非全黑）
def is_valid_image(image, threshold=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_image)
    return mean_intensity > threshold

# 检查照明条件是否正常
def is_valid_illumination(image, min_brightness=30, max_brightness=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_image)
    return min_brightness <= mean_brightness <= max_brightness

# 控制器运动函数
def move_controller(direction, step_size):
    logging.info(f"在方向{direction}上以步长{step_size}移动控制器")
    controller.move_by(axis=1, steps=step_size * direction, addr=None)

# 捕获桌面线程
def capture_thread(capture_area, image_queue, stop_event):
    logging.info("捕获线程开始")
    while not stop_event.is_set():
        start_time = time.time()
        frame = capture_screen(capture_area)
        capture_time = time.time() - start_time

        if not is_valid_image(frame):
            logging.warning("检测到无效图像,跳过此帧.")
            continue

        # 检查队列状态
        if image_queue.full():
            logging.warning("图像队列已满,丢弃最旧的帧。")
            image_queue.get()  # 如果队列已满，丢弃最早的数据
        image_queue.put(frame)
        logging.info(f"捕获线程-帧处理时间为{time.time() - start_time:.4f}秒")
        logging.info(f"添加帧后的队列大小:{image_queue.qsize()}")
        time.sleep(0.15)  # 控制捕获频率
    logging.info("捕获线程停止")

# 检测+跟踪+聚焦判断+控制器运动线程
def process_thread(image_queue, stop_event, target_track_id):
    logging.info("处理线程开始")

    # 初始化 YOLO 模型
    model = YOLO("best.pt")

    # 初始化 DeepSORT 跟踪器
    deepsort = DeepSort(model_path=rid_model_path,
                        max_dist=0.2, min_confidence=0.3,
                        nms_max_overlap=0.5, max_iou_distance=0.7,
                        max_age=500, n_init=20, nn_budget=100, use_cuda=True)

    # 控制器运动参数
    step_size = 100  # 每次移动的步数
    direction = 1  # 初始运动方向（1 表示正向，-1 表示反向）
    last_focus_score = 0  # 上一次的聚焦分数

    start_time = time.time()  # 开始调整聚焦的时间
    total_time = 0  # 总聚焦时间
    frame_time = 0  # 每帧处理时间
    move_time = 0  # 每次运动时间
    focus_score_time = 0  # 初始化 focus_score_time

    while not stop_event.is_set():
        try:
            # 获取图像
            if image_queue.empty():
                logging.warning("图像队列为空，正在等待新的帧图像···")
                time.sleep(0.01)  # 避免过度占用CPU
                continue

            # 检查队列状态
            logging.info(f"处理线程-获取帧之前的队列大小：{image_queue.qsize()}")
            start_time = time.time()
            frame = image_queue.get()
            queue_time = time.time() - start_time
            logging.info(f"处理线程-获取帧后的队列大小：{image_queue.qsize()}")

            # 检查照明条件是否正常
            if not is_valid_illumination(frame):
                logging.warning("检测到无效照明，跳过此帧。")
                continue

            # 记录每帧处理开始时间
            frame_start_time = time.time()

            # 检测目标
            results = model(frame)
            detection_time = time.time() - frame_start_time
            
            # 提取检测结果
            xywhs = []
            confss = []
            clss = []
            for result in results:
                for *xywh, r, conf, cls in result.obb.data.cpu().numpy():
                    x, y, w, h = map(int, xywh)
                    xywhs.append([int(x), int(y), int(w), int(h)])
                    confss.append(conf)
                    clss.append(cls)
            # 如果没有检测到任何目标，跳过当前帧的跟踪处理
            if not xywhs:
                logging.warning("没有检测结果")
                continue
            # 将列表转换为 numpy 数组
            xywhs = np.array(xywhs)
            confss = np.array(confss)
            clss = np.array(clss)

            # 更新跟踪器
            tracks_start_time = time.time()
            tracks = deepsort.update(xywhs, confss, clss, frame)
            tracking_time = time.time() - tracks_start_time
            
            # 绘制跟踪框
            for track in tracks[0]:
                x1, y1, x2, y2, class_id, track_id = track
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # 如果是用户指定的编号，进行检测框膨胀和聚焦计算
                if track_id == target_track_id:
                    # 检测框膨胀
                    expand_ratio = 1.5  # 膨胀比例
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    width, height = x2 - x1, y2 - y1
                    new_width, new_height = int(width * expand_ratio), int(height * expand_ratio)

                    # 计算新的边界框坐标
                    new_x1 = max(0, center_x - new_width // 2)
                    new_y1 = max(0, center_y - new_height // 2)
                    new_x2 = min(frame.shape[1], center_x + new_width // 2)
                    new_y2 = min(frame.shape[0], center_y + new_height // 2)

                    # 提取膨胀框内的图像区域
                    cropped_image = frame[new_y1:new_y2, new_x1:new_x2]

                    # 计算聚焦分数
                    focus_score_start_time = time.time()
                    focus_score = calculate_focus_score(cropped_image)
                    focus_score_time = time.time() - focus_score_start_time

                    # 绘制膨胀框
                    is_focused_result = is_focused(cropped_image)

                    cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 1)
                    cv2.putText(frame,
                                f"id:{track_id} (focus score:{focus_score:.2f},focused:{is_focused_result})",
                                (new_x1, new_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 根据聚焦分数调整控制器运动
                    if not is_focused_result:
                        logging.info(f"目标{track_id}不聚焦.调整控制器...")
                        if focus_score < last_focus_score:
                            # 如果聚焦分数变小，反向运动
                            direction = -direction
                        move_start_time = time.time() # 记录运动开始时间
                        #move_controller(direction, step_size)
                        move_time = time.time() - move_start_time  # 记录运动时间
                    else:
                        logging.info(f"目标{target_track_id}聚焦.")
                        #controller.stop(axis=1, addr=None)
                        total_time = time.time() - start_time  # 记录总聚焦时间
                        logging.info(f"总聚焦时间：{total_time:.2f}s")
                        # 清空队列中的所有图像数据
                        while not image_queue.empty():
                            image_queue.get()
                        time.sleep(0.5)  # 等待一段时间后重新开始聚焦
                        continue  # 重新开始聚焦流程

                    # 更新上一次的聚焦分数
                    last_focus_score = focus_score

            # 计算并显示亮度值
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_image)
            cv2.putText(frame, f"Brightness: {mean_brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 记录每帧处理时间
            frame_time = time.time() - frame_start_time
            logging.info(f"帧处理时间{frame_time:.4f}s")
            logging.info(f"检测时间:{detection_time:.4f}s")
            logging.info(f"跟踪时间:{tracking_time:.4f}s")
            logging.info(f"聚焦分数计算时间：{focus_score_time:.4f}s")
            logging.info(f"控制器运动时间：{move_time:.4f}s")

            # 显示实时结果
            cv2.imshow('YOLO Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()  # 停止所有线程
                break

        except Exception as e:
            logging.error(f"处理帧时出错:{e}")
            time.sleep(1)  # 出现异常时暂停1秒，避免频繁报错

    logging.info("处理线程已停止.")
    cv2.destroyAllWindows()

# 主程序
def main():
    # 初始化控制器
    #init_controller()

    # 捕获屏幕的区域 (x, y, width, height)
    capture_area = (0, 70, 1112, 886)

    # 创建线程安全的队列
    image_queue = queue.Queue(maxsize=10)  # 最多存储10帧图像

    # 创建停止事件
    stop_event = threading.Event()

    # 用户输入指定编号
    target_track_id = int(input("请输入需要关注的目标编号："))

    # 创建线程
    capture_thread_obj = threading.Thread(target=capture_thread, args=(capture_area, image_queue, stop_event))
    process_thread_obj = threading.Thread(target=process_thread, args=(image_queue, stop_event, target_track_id))
    
    # 启动线程
    capture_thread_obj.start()
    process_thread_obj.start()

    # 等待线程结束
    capture_thread_obj.join()
    process_thread_obj.join()

    logging.info("程序结束.")

if __name__ == "__main__":
    main()