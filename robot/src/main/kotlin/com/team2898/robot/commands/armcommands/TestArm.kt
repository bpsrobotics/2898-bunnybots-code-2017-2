package com.team2898.robot.commands.armcommands

import com.team2898.engine.logging.LogLevel
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.reflectLocation
import com.team2898.robot.subsystems.Elbow
import edu.wpi.first.wpilibj.Timer
import edu.wpi.first.wpilibj.command.Command

class TestArm: Command() {
    var startTime = 0.0
    override fun initialize() {
        startTime = Timer.getFPGATimestamp()
    }
    override fun execute() {
        if (isFinished()) return
        Logger.logInfo(reflectLocation(), LogLevel.DEBUG, "TestArm execute loop")
        //Elbow.motor.set(0.3)
        Logger.logInfo(reflectLocation(), LogLevel.DEBUG, "Finished TestArm execute loop")
    }
    override fun end() {
        //Elbow.motor.setOpen(0.0)
    }

    override fun isFinished(): Boolean {
        println (Timer.getFPGATimestamp() - startTime)
        return (Timer.getFPGATimestamp() - startTime) > 0.5
    }
}