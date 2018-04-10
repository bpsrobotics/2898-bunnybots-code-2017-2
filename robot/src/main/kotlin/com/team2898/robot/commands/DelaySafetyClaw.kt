package com.team2898.robot.commands

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.config.AutoConf.CLAW_DEPLOY_SAFETY_THRESHOLD
import com.team2898.robot.subsystems.Elbow
import edu.wpi.first.wpilibj.command.Command

class DelaySafetyClaw: Command() {
    override fun isFinished(): Boolean {
        val value  = Math.abs(Elbow.currentPose.degrees())
        println("Delay safety claw: $value, $CLAW_DEPLOY_SAFETY_THRESHOLD")
        return value < CLAW_DEPLOY_SAFETY_THRESHOLD
    }
}