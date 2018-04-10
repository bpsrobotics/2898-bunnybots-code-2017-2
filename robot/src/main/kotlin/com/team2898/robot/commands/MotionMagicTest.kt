package com.team2898.robot.commands

import com.team2898.engine.motion.DriveSignal
import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.command.Command

class MotionMagicTest(): Command() {
    override fun initialize() {
        Drivetrain.leftMaster.position = 0.0
        Drivetrain.rightMaster.position = 0.0
        Drivetrain.leftMaster.encPosition = 0
        Drivetrain.rightMaster.encPosition = 0
        Drivetrain.controlMode = Drivetrain.ControlModes.MOTION_MAGIC
    }

    override fun execute() {
        Drivetrain.motionMagicPositions = DriveSignal(5.0,5.0)
    }

    override fun isFinished(): Boolean {
        return false
    }
}