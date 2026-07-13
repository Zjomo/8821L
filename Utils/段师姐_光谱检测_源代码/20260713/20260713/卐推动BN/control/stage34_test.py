from pylablib.devices import Thorlabs
stage = Thorlabs.KinesisPiezoMotor("97101208")
stage.setup_drive(max_voltage=100, velocity=500, acceleration=500, channel=3)
stage.move_by(distance=-100, channel=3)
stage.move_by(distance=100, channel=3)
stage.setup_drive(max_voltage=100, velocity=500, acceleration=500, channel=4)
stage.move_by(distance=-100, channel=4) 
stage.move_by(distance=100, channel=4)
stage.close()