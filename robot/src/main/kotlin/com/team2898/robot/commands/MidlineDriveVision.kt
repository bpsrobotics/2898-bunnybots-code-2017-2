package com.team2898.robot.commands

import com.sun.xml.internal.bind.api.impl.NameConverter
import com.team2898.engine.controlLoops.AsyncPID
import com.team2898.engine.controlLoops.StandardPID
import com.team2898.engine.motion.CheesyDrive
import com.team2898.engine.motion.DriveSignal
import com.team2898.robot.VisionComms
import com.team2898.robot.config.AutoConf.VISION_DISTANCE
import com.team2898.robot.subsystems.Drivetrain
import com.team2898.robot.subsystems.Elbow
import com.team2898.robot.subsystems.Navx
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.command.Command
import edu.wpi.first.wpilibj.command.PIDCommand

class MidlineDriveVision : Command() {
//    override fun usePIDOutput(output: Double) {
//        Drivetrain.openLoopPower = CheesyDrive.updateCheesy(
//                -output,
//                0.20,
//                false,
//                true
//        )
//        //println("Target gyro angle: $targetGyroAngle\nActual gyro angle: ${Navx.yaw}\nPid output: ${output}\nCamera angle: ${VisionComms.bucketAngle}\nDistance: $targetDistance")
//        println("PID output: ${output}\nBucket angle: ${VisionComms.bucketAngle}\nBucket distance: ${targetDistance}")
//        println("isEnabled: ${pidController.isEnabled}\n")
//    }

//    override fun returnPIDInput(): Double {
//        pidController.setpoint = ((Navx.yaw - VisionComms.bucketAngle) + 360.0) % 360.0
//        return Navx.yaw
//    }

    var targetGyroAngle: Double = 0.0
        set(value) {
            field = value
//            pidController.setpoint = field
        }

    init {
//        pidController.setInputRange(0.0, 359.0)
//        pidController.setOutputRange(-1.0, 1.0)
//        pidController.setPercentTolerance(0.1)
    }

    val targetDistance: Double
        get() = VisionComms.bucketDistance

    override fun initialize() {
        Drivetrain.gearMode = Drivetrain.GearModes.LOW
        Drivetrain.controlMode = Drivetrain.ControlModes.OPEN_LOOP
        Wrist.targetPose = Wrist.currentPose
        Elbow.targetPose = Elbow.targetPose
        if (!Drivetrain.loop.m_job.isActive) Drivetrain.loop.start()
        targetGyroAngle = 0.0
//        pidController.enable()
    }

    override fun end() {
//        pidController.disable()
        Drivetrain.openLoopPower = DriveSignal.BRAKE
        Drivetrain.openLoopPower = DriveSignal(0.0, 0.0)
        println("ending")
    }

    override fun execute() {
        val setpoint = ((Navx.yaw - VisionComms.bucketAngle) + 360.0) % 360.0

        var error = setpoint - Navx.yaw

        if ((Math.abs(error) > (359.9- 0.0) / 2)) {
            error = if (error > 0) error - 359.9 + 0.0 else error + 359.9 + 0.0
        }

        Drivetrain.openLoopPower = CheesyDrive.updateCheesy(
                - error * 0.025,
                //-VisionComms.bucketAngle * 0.005,
                0.10,
                false,
                true
        )
        println("PID value: ${-error * 0.025}")
        println("angle: ${VisionComms.bucketAngle}\n")
    }

    override fun isFinished(): Boolean {
        return targetDistance <= 20.0
        //return false
    }
}