package com.team2898.robot.commands

import com.team2898.engine.motion.CheesyDrive
import com.team2898.robot.subsystems.Drivetrain
import com.team2898.robot.subsystems.Navx
import edu.wpi.first.wpilibj.command.PIDCommand

class StraightAuto(val distance: Double): PIDCommand(0.01, 0.001, 0.1) {
    override fun returnPIDInput(): Double {
        return Navx.yaw
    }

    var turn: Double = 0.0

    override fun usePIDOutput(output: Double) {
        turn= output
    }

    override fun initialize() {
        pidController.setContinuous(true)
        pidController.setInputRange(0.0, 359.9)
        pidController.setpoint = 0.0
        pidController.setOutputRange(-1.0, 0.0)
        Drivetrain.gearMode = Drivetrain.GearModes.LOW
        Drivetrain.controlMode = Drivetrain.ControlModes.OPEN_LOOP
    }

    override fun execute () {
        Drivetrain.openLoopPower = CheesyDrive.updateCheesy(
                turn, 0.2, false, true
        )
    }

    override fun isFinished(): Boolean {
        return ((Math.abs(Drivetrain.encPos.x) + Math.abs(Drivetrain.encPos.y))/2.0) > distance
    }

}