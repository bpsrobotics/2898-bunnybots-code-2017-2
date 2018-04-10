package com.team2898.robot.subsystems

import edu.wpi.first.wpilibj.DoubleSolenoid
import com.team2898.robot.config.RobotMap.*

object Claw {
    val clawValve = DoubleSolenoid(CLAW_SOLENOID_FORWARD_ID, CLAW_SOLENOID_REVERSE_ID)

    enum class ClawState {
        OPEN, CLOSED
    }

    var clawState: ClawState = ClawState.CLOSED
        set(value) {
            if (field != value) {
                field = value
                when (clawState) {
                    ClawState.CLOSED -> clawValve.set(DoubleSolenoid.Value.kForward)
                    ClawState.OPEN -> clawValve.set(DoubleSolenoid.Value.kReverse)
                }
            }
        }

    init {
        clawValve.set(DoubleSolenoid.Value.kForward)
    }
}