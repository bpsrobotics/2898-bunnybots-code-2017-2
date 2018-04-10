package com.team2898.robot.commands

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.config.AutoConf.ELBOW_MAX_POSE_ERROR
import com.team2898.robot.config.AutoConf.WRIST_MAX_POSE_ERROR
import com.team2898.robot.subsystems.Elbow
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.command.Command

class ArmPose(val elbowPose: Rotation2d, val wristPose: Rotation2d, val waitForElbow: Boolean = true, val waitForWrist: Boolean = true) : Command() {
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

    override fun isFinished(): Boolean {
        return ((Math.abs(Elbow.motor.closedLoopError) < ELBOW_MAX_POSE_ERROR) || !waitForElbow) &&
                ((Math.abs(Wrist.motor.closedLoopError) < WRIST_MAX_POSE_ERROR) || !waitForWrist)
    }
}