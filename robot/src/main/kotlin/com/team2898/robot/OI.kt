package com.team2898.robot

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.config.OIConf.*
import com.team2898.robot.config.TeleopConfig.armControllerElbowOffset
import com.team2898.robot.config.TeleopConfig.armControllerTicsToDegreesConstant
import com.team2898.robot.config.TeleopConfig.armControllerWristOffset
import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.Joystick
import edu.wpi.first.wpilibj.networktables.NetworkTable

object OI {

    val networkTable = NetworkTable.getTable("controller")

    fun deadzone(value: Double): Double {
        if (Math.abs(value) < DEADZONE_THRESHOLD) return 0.0
        return value
    }

    fun cube(value: Double): Double = Math.pow(value, 3.0)
    fun square(value: Double): Double = Math.pow(value, 2.0) * if (value > 0) 1 else -1

    fun process(value: Double, deadzone: Boolean = true, cube: Boolean = false, square: Boolean = false): Double {
        var localValue = value
        if (deadzone)
            localValue = deadzone(value)
        if (cube) {
            localValue = cube(localValue)
        } else if (square) {
            localValue = square(localValue)
        }

        return localValue
    }

    val driverController: Joystick = Joystick(0)
    val operatorController: Joystick = Joystick(1)

    val armControllerWristEncoder: Double
        get() =
            (-networkTable.getNumber("enc1", 0.0) * armControllerTicsToDegreesConstant - armControllerWristOffset) % 360

    val armControllerElbowEncoder: Double
        get() =
            (-networkTable.getNumber("enc2", 0.0) * armControllerTicsToDegreesConstant + armControllerElbowOffset) % 360

    val throttle
        get() = process(driverController.getRawAxis(1), square = true)
    val turn
        get() = process(driverController.getRawAxis(4), square = Drivetrain.gearMode == Drivetrain.GearModes.HIGH)
    val quickTurn: Boolean
        get() = process(Math.max(driverController.getRawAxis(2), driverController.getRawAxis(3))) != 0.0
    val leftTrigger
        get() = process(driverController.getRawAxis(2), square = true)
    val rightTrigger
        get() = process(driverController.getRawAxis(3), square = true)
    val brake
        get() = driverController.getRawButton(8) || driverController.getRawButton(9)
    val lowGear
        get() = driverController.getRawButton(5)
    val highGear
        get() = driverController.getRawButton(6)

    val operatorLeftX
        get() = process(operatorController.getRawAxis(0))
    val operatorLeftY
        get() = process(operatorController.getRawAxis(1))
    val operatorRightX
        get() = process(operatorController.getRawAxis(4))
    val operatorRightY
        get() = process(operatorController.getRawAxis(5))

    val claw
        get() = driverController.getRawButton(2) || driverController.getRawButton(1) ||
                operatorController.getRawButton(2) || operatorController.getRawButton(1) ||
                operatorController.getRawButton(3) || operatorController.getRawButton(4)

    val clawOpen
        get() = driverController.getRawButton(0)
    val clawClose
        get() = driverController.getRawButton(1)
}
