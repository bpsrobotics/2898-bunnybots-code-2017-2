package com.team2898.robot.commands

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.subsystems.Claw
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.command.Command
import edu.wpi.first.wpilibj.command.InstantCommand

class CloseClaw: InstantCommand() {
    override fun initialize() {
        Claw.clawState = Claw.ClawState.CLOSED
    }
}