package com.team2898.robot.commands.armcommands

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.config.AutoConf.KINECT_POSE_ELBOW_DEG
import com.team2898.robot.subsystems.Elbow
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.command.InstantCommand

class KinectPose: InstantCommand() {
    override fun initialize() {
        Elbow.targetPose = Rotation2d.createFromRadians(KINECT_POSE_ELBOW_DEG)
        Wrist.targetPose = Rotation2d(1.0, 0.0)
    }
}