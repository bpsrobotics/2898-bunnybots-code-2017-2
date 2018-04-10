package com.team2898.robot.commands

import com.team2898.engine.controlLoops.AsyncPID
import com.team2898.engine.extensions.Vector2D.*
import com.team2898.engine.motion.CheesyDrive
import com.team2898.robot.VisionComms
import com.team2898.robot.config.AutoConf.VISION_DISTANCE
import com.team2898.robot.subsystems.Drivetrain
import com.team2898.robot.subsystems.Navx
import edu.wpi.first.wpilibj.command.Command
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D

class MidlineDriveFinalApproach : Command() {

    var startPos = Vector2D(0.0, 0.0)

    val pidController = AsyncPID(
            execHz = 50.0,
            Kp = 0.0185,//TODO
            Ki = 0.0,//TODO
            Kd = 0.0 //TODO
    )

    init {
        pidController.apply {
            continuous = true
            setInputRange(0.0, 359.9) //TODO
            setOutputRange(-1.0, 1.0) //TODO
            getSensorInput = { Navx.yaw }
            useControllerOutput = { output ->
                Drivetrain.openLoopPower = CheesyDrive.updateCheesy(
                        output,
                        0.25,
                        false,
                        true
                )
            }
        }
    }

    override fun initialize() {
        // triggers lazy and sets startPos to drivetrain enc pos
        startPos = Drivetrain.encPos
        pidController.setpoint = Navx.yaw
        pidController.start()
    }

    override fun execute() {

    }

    override fun end() {
        pidController.stop()
    }

    override fun isFinished(): Boolean {
        return ((Drivetrain.encPos - startPos) / 2.0).l1 * 6 * Math.PI >= VISION_DISTANCE * 12
        //return (findAve(Drivetrain.encPos) - findAve(startPosition)) * 6 * Math.PI > 7 * 12
    }

    private fun findAve(pair: Pair<Double, Double>): Double {
        return (pair.first + pair.second) / 2
    }
}