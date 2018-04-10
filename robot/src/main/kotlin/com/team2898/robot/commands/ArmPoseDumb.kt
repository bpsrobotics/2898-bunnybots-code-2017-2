package com.team2898.robot.commands

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.subsystems.Elbow
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.command.TimedCommand

class ArmPoseDumb(val elbowPose: Rotation2d, val wristPose: Rotation2d, val time: Double): TimedCommand(time) {
    override fun initialize() {
        setPoses()
    }

    fun setPoses() {
        Wrist.targetPose = wristPose
        Elbow.targetPose = elbowPose
    }

    override fun execute() {
        setPoses()
    }
}