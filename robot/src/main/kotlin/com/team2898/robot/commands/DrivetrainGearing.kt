package com.team2898.robot.commands

import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.command.InstantCommand

class DtLowGear(): InstantCommand() {
    override fun initialize() {
        Drivetrain.gearMode = Drivetrain.GearModes.LOW
    }
}

class DtHighGear(): InstantCommand() {
    override fun initialize() {
        Drivetrain.gearMode = Drivetrain.GearModes.HIGH
    }
}
