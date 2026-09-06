from pylablib.devices import Newport
import time

# 查看设备数
Newport.get_usb_devices_number_picomotor()
stage1 = Newport.Picomotor8742()
print('设备连接成功')
stage1.close()

# 设备运动封装
class PicoMotor8742Controller:
    """
    Newport Picomotor 8743 控制封装（pylablib）
    - 轴号: 1~4
    - 常用：move_by(相对)、move_to(绝对)、get_position、wait_move、stop、setup_velocity、jog
    """

    def __init__(self, conn=0, *, backend="auto", timeout=5.0, multiaddr=False, scan=True):
        self.conn = conn
        self.backend = backend
        self.timeout = timeout
        self.multiaddr = multiaddr
        self.scan = scan
        self.dev = None

    # ---------- 连接/关闭 ----------
    @staticmethod
    def usb_device_count() -> int:
        return Newport.get_usb_devices_number_picomotor()

    """ 连接设备 """
    def open(self):
        # conn: USB 用 index(0,1,2...)；Ethernet 用 IP/hostname（如 "8742-12345"）
        self.dev = Newport.Picomotor8742(
            self.conn,
            backend=self.backend,
            timeout=self.timeout,
            multiaddr=self.multiaddr,
            scan=self.scan
        )
        return self

    """ 关闭设备 """
    def close(self):
        if self.dev is not None:
            self.dev.close()
            self.dev = None

    """ 写入enter """
    def __enter__(self):
        return self.open()

    """ 写入exit """
    def __exit__(self, exc_type, exc, tb):
        self.close()

    """ 检查连接情况  """
    def _check(self):
        if self.dev is None:
            raise RuntimeError(".")
    # 最好补充一个逻辑

    """ 获取设备ID """
    def get_id(self, *, addr=None):
        self._check()
        return self.dev.get_id(addr=addr)

    """ 获取已有轴的列表 """
    def axes(self):
        self._check()
        # 文档提到可用 get_all_axes() 获取轴列表
        return self.dev.get_all_axes()

    """ 返回 控制器累计的步数"""
    def get_pos(self, axis=1, *, addr=None) -> int:
        self._check()
        # get_position(axis="all") / 指定 axis
        return self.dev.get_position(axis=axis, addr=addr)    

    """ 把当前位置设为参考零点（不实际移动）""" 
    def set_zero_here(self, axis=1, position=0, *, addr=None) -> int:
        self._check()
        return self.dev.set_position_reference(axis=axis, position=position, addr=addr)  
        
    """ 相对移动（推荐）"""
    def move_rel(self, axis=1, steps=100, wait=True, *, addr=None):
        self._check()
        self.dev.move_by(axis=axis, steps=steps, addr=addr)  
        if wait:
            self.dev.wait_move(axis=axis, addr=addr) 
            
    """ 绝对移动（开环计步，可能漂移）"""
    def move_abs(self, axis=1, position=0, wait=True, *, addr=None):
        self._check()
        self.dev.move_to(axis=axis, position=position, addr=addr) 
        if wait:
            self.dev.wait_move(axis=axis, addr=addr) 

    ''' 判断是否在移动 '''
    def is_moving(self, axis=1, *, addr=None) -> bool:
        self._check()
        return self.dev.is_moving(axis=axis, addr=addr) 

    ''' 等电机把当前这次运动彻底走完，再继续往下执行程序 '''
    def wait(self, axis=1, *, addr=None):
        self._check()
        self.dev.wait_move(axis=axis, addr=addr)  

    """停止：immediate=True 为急停（且必须 axis='all'）"""
    def stop(self, axis="all", immediate=False, *, addr=None):
        self._check()
        self.dev.stop(axis=axis, immediate=immediate, addr=addr) 

    """返回 (speed, accel)，单位 steps/s, steps/s^2"""
    # ---------- 速度/加速度 ----------
    def get_vel(self, axis=1, *, addr=None):
        self._check()
        return self.dev.get_velocity_parameters(axis=axis, addr=addr) 

    def set_vel(self, axis=1, speed=None, accel=None, *, addr=None):
        """设置速度/加速度；传 None 表示不改"""
        self._check()
        return self.dev.setup_velocity(axis=axis, speed=speed, accel=accel, addr=addr) 

    # ---------- 点动 jog ----------
    def jog(self, axis=1, direction="+", *, addr=None):
        """持续点动：direction 为 '+' 或 '-'；需要 stop() 停下来"""
        self._check()
        dir_flag = (direction == "+")
        self.dev.jog(axis=axis, direction=dir_flag, addr=addr) 

     # ---------- 发送原始 NMCL 命令 ----------
    def raw_query(self, cmd: str, *, axis=None, addr=None):
        """
        直接发送 NMCL 命令；cmd 里不要带轴号前缀（axis 参数会自动加到命令前）
        例如：raw_query("TP?", axis=1)  -> 实际发送 "1TP?"
             raw_query("SM")           -> 实际发送 "SM"
        """
        self._check()
        return self.dev.query(cmd, axis=axis, addr=addr)
  

import sys
import time
from typing import Optional, Dict, List
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


# 单个轴的控制面板
class MotorAxisWidget(QWidget):
    def __init__(self, axis_num: int, controller: PicoMotor8742Controller, parent=None):
        super().__init__(parent)
        self.axis = axis_num
        self.controller = controller
        self.init_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_position)
        self.update_timer.start(500)  # 500ms更新一次位置
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 轴标题
        title_label = QLabel(f"<h3>轴 {self.axis}</h3>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 当前位置显示
        pos_group = QGroupBox("当前位置")
        pos_layout = QVBoxLayout()
        self.pos_label = QLabel("0")
        self.pos_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.pos_label.setAlignment(Qt.AlignCenter)
        pos_layout.addWidget(self.pos_label)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # 速度设置
        vel_group = QGroupBox("速度参数")
        vel_layout = QGridLayout()
        
        vel_layout.addWidget(QLabel("速度 (steps/s):"), 0, 0)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 5000)
        self.speed_spin.setValue(200)
        vel_layout.addWidget(self.speed_spin, 0, 1)
        
        vel_layout.addWidget(QLabel("加速度 (steps/s²):"), 1, 0)
        self.accel_spin = QSpinBox()
        self.accel_spin.setRange(1, 10000)
        self.accel_spin.setValue(2000)
        vel_layout.addWidget(self.accel_spin, 1, 1)
        
        self.set_vel_btn = QPushButton("设置速度")
        self.set_vel_btn.clicked.connect(self.set_velocity)
        vel_layout.addWidget(self.set_vel_btn, 2, 0, 1, 2)
        
        vel_group.setLayout(vel_layout)
        layout.addWidget(vel_group)
        
        # 相对移动
        rel_group = QGroupBox("相对移动")
        rel_layout = QGridLayout()
        
        rel_layout.addWidget(QLabel("步数:"), 0, 0)
        self.rel_steps_spin = QSpinBox()
        self.rel_steps_spin.setRange(-10000, 10000)
        self.rel_steps_spin.setValue(100)
        rel_layout.addWidget(self.rel_steps_spin, 0, 1)
        
        self.move_rel_btn = QPushButton("正向移动")
        self.move_rel_btn.clicked.connect(lambda: self.move_relative(self.rel_steps_spin.value()))
        rel_layout.addWidget(self.move_rel_btn, 1, 0)
        
        self.move_rel_neg_btn = QPushButton("反向移动")
        self.move_rel_neg_btn.clicked.connect(lambda: self.move_relative(-self.rel_steps_spin.value()))
        rel_layout.addWidget(self.move_rel_neg_btn, 1, 1)
        
        rel_group.setLayout(rel_layout)
        layout.addWidget(rel_group)
        
        # 绝对移动
        abs_group = QGroupBox("绝对移动")
        abs_layout = QGridLayout()
        
        abs_layout.addWidget(QLabel("目标位置:"), 0, 0)
        self.abs_pos_spin = QSpinBox()
        self.abs_pos_spin.setRange(-100000, 100000)
        self.abs_pos_spin.setValue(0)
        abs_layout.addWidget(self.abs_pos_spin, 0, 1)
        
        self.move_abs_btn = QPushButton("绝对移动")
        self.move_abs_btn.clicked.connect(self.move_absolute)
        abs_layout.addWidget(self.move_abs_btn, 1, 0, 1, 2)
        
        abs_group.setLayout(abs_layout)
        layout.addWidget(abs_group)
        
        # 点动控制
        jog_group = QGroupBox("点动控制")
        jog_layout = QHBoxLayout()
        
        self.jog_plus_btn = QPushButton("正向点动")
        self.jog_plus_btn.pressed.connect(lambda: self.start_jog("+"))
        self.jog_plus_btn.released.connect(self.stop_jog)
        jog_layout.addWidget(self.jog_plus_btn)
        
        self.jog_minus_btn = QPushButton("反向点动")
        self.jog_minus_btn.pressed.connect(lambda: self.start_jog("-"))
        self.jog_minus_btn.released.connect(self.stop_jog)
        jog_layout.addWidget(self.jog_minus_btn)
        
        jog_group.setLayout(jog_layout)
        layout.addWidget(jog_group)
        
        # 零点设置
        zero_btn = QPushButton("设置当前位置为零点")
        zero_btn.clicked.connect(self.set_zero)
        layout.addWidget(zero_btn)
        
        # 停止按钮
        stop_btn = QPushButton("停止运动")
        stop_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        stop_btn.clicked.connect(self.stop_movement)
        layout.addWidget(stop_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_position(self):
        try:
            pos = self.controller.get_pos(self.axis)
            self.pos_label.setText(str(pos))
        except Exception as e:
            self.pos_label.setText("Error")
            print(f"更新位置时出错: {e}")
    
    def set_velocity(self):
        try:
            self.controller.set_vel(
                axis=self.axis,
                speed=self.speed_spin.value(),
                accel=self.accel_spin.value()
            )
            QMessageBox.information(self, "成功", "速度参数已设置")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置速度失败: {e}")
    
    def move_relative(self, steps: int):
        try:
            self.controller.move_rel(axis=self.axis, steps=steps, wait=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"移动失败: {e}")
    
    def move_absolute(self):
        try:
            self.controller.move_abs(
                axis=self.axis,
                position=self.abs_pos_spin.value(),
                wait=True
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"绝对移动失败: {e}")
    
    def start_jog(self, direction: str):
        try:
            self.controller.jog(axis=self.axis, direction=direction)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"点动失败: {e}")
    
    def stop_jog(self):
        try:
            self.controller.stop(axis=self.axis)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"停止失败: {e}")
    
    def set_zero(self):
        try:
            self.controller.set_zero_here(axis=self.axis)
            QMessageBox.information(self, "成功", f"轴 {self.axis} 零点已设置")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置零点失败: {e}")
    
    def stop_movement(self):
        try:
            self.controller.stop(axis=self.axis, immediate=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"停止失败: {e}")


# 主控制界面
class PicomotorControlGUI(QMainWindow):
    """主控制界面"""
    def __init__(self):
        super().__init__()
        self.controller: Optional[PicoMotor8742Controller] = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Newport Picomotor 8742 控制器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("<h1>Newport Picomotor 8742 控制器</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 连接状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("未连接")
        self.status_bar.addWidget(self.status_label)
        
        # 连接控制区域
        conn_group = QGroupBox("设备连接")
        conn_layout = QHBoxLayout()
        
        self.usb_count_label = QLabel("检测到 USB 设备: 0")
        conn_layout.addWidget(self.usb_count_label)
        
        conn_layout.addWidget(QLabel("设备索引:"))
        self.device_index_spin = QSpinBox()
        self.device_index_spin.setRange(0, 10)
        self.device_index_spin.setValue(0)
        conn_layout.addWidget(self.device_index_spin)
        
        self.connect_btn = QPushButton("连接设备")
        self.connect_btn.clicked.connect(self.connect_device)
        conn_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开连接")
        self.disconnect_btn.clicked.connect(self.disconnect_device)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn)
        
        self.refresh_btn = QPushButton("刷新设备列表")
        self.refresh_btn.clicked.connect(self.refresh_usb_count)
        conn_layout.addWidget(self.refresh_btn)
        
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)
        
        # 设备信息
        self.info_group = QGroupBox("设备信息")
        self.info_layout = QGridLayout()
        self.info_labels = {}
        
        self.info_labels['id'] = QLabel("ID: N/A")
        self.info_layout.addWidget(self.info_labels['id'], 0, 0)
        
        self.info_labels['axes'] = QLabel("可用轴: N/A")
        self.info_layout.addWidget(self.info_labels['axes'], 0, 1)
        
        self.info_group.setLayout(self.info_layout)
        self.info_group.setVisible(False)
        main_layout.addWidget(self.info_group)
        
        # 多轴控制面板
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 日志区域
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        self.clear_log_btn = QPushButton("清除日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_btn)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 初始化
        self.refresh_usb_count()
        
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        
    def refresh_usb_count(self):
        """刷新USB设备数量"""
        try:
            count = PicoMotor8742Controller.usb_device_count()
            self.usb_count_label.setText(f"检测到 USB 设备: {count}")
        except Exception as e:
            self.usb_count_label.setText(f"检测失败: {e}")
            
    def connect_device(self):
        """连接设备"""
        try:
            device_index = self.device_index_spin.value()
            self.log_message(f"正在连接设备 {device_index}...")
            
            self.controller = PicoMotor8742Controller(conn=device_index)
            self.controller.open()
            
            # 获取设备信息
            device_id = self.controller.get_id()
            available_axes = self.controller.axes()
            
            # 更新UI
            self.info_labels['id'].setText(f"ID: {device_id}")
            self.info_labels['axes'].setText(f"可用轴: {available_axes}")
            self.info_group.setVisible(True)
            
            # 创建轴控制面板
            self.tab_widget.clear()
            for axis in available_axes:
                tab = MotorAxisWidget(axis, self.controller)
                self.tab_widget.addTab(tab, f"轴 {axis}")
            
            # 更新按钮状态
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.device_index_spin.setEnabled(False)
            
            self.status_label.setText(f"已连接 - {device_id}")
            self.log_message(f"设备连接成功: {device_id}")
            
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"无法连接设备: {e}")
            self.log_message(f"连接失败: {e}")
            
    def disconnect_device(self):
        """断开设备连接"""
        try:
            if self.controller:
                self.controller.close()
                self.controller = None
                
                # 更新UI
                self.tab_widget.clear()
                self.info_group.setVisible(False)
                
                self.connect_btn.setEnabled(True)
                self.disconnect_btn.setEnabled(False)
                self.device_index_spin.setEnabled(True)
                
                self.status_label.setText("未连接")
                self.log_message("设备已断开连接")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"断开连接时出错: {e}")
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.disconnect_device()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，更现代
    
    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
    """)
    
    window = PicomotorControlGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()