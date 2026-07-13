from pylablib.devices import Thorlabs
stage = Thorlabs.KinesisPiezoMotor("97101208")
stage.setup_drive(max_voltage=100, velocity=500, acceleration=500, channel=1)
stage.move_by(distance=-1000, channel=1)#右上
#stage.move_by(distance=1000, channel=1)# 左下
#stage.setup_drive(max_voltage=100, velocity=500, acceleration=500, channel=2)
#stage.move_by(distance=-1000, channel=2) #左上
#stage.move_by(distance=1000, channel=2)#右下
stage.close()