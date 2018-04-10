package com.team2898.robot.commands

import com.team2898.engine.motion.DriveSignal
import com.team2898.robot.config.AutoConf.MOTIONMAGIC_ERROR_THRESHOLD
import com.team2898.robot.config.DrivetrainConf.LEFT_MM_STRAIGHT_ADJ_FACTOR
import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.command.Command

class DriveStraightDistance(val distance: Double): Command(){
    override fun initialize() {
        Drivetrain.controlMode = Drivetrain.ControlModes.MOTION_MAGIC
    }

    override fun execute() {
        Drivetrain.motionMagicPositions = DriveSignal((distance* LEFT_MM_STRAIGHT_ADJ_FACTOR)/4, distance/4)
    }

    override fun isFinished(): Boolean {
        return Math.abs(Drivetrain.leftMaster.closedLoopError) < MOTIONMAGIC_ERROR_THRESHOLD &&
                Math.abs(Drivetrain.rightMaster.closedLoopError) < MOTIONMAGIC_ERROR_THRESHOLD
    }
}
