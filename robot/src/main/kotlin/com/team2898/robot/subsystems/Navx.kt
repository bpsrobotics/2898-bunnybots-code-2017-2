package com.team2898.robot.subsystems

import com.kauailabs.navx.frc.AHRS
import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.kinematics.Rotation2d
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.reflectLocation
import com.team2898.engine.signalProcessing.MovingAverageFilter
import com.team2898.engine.types.CircularArray
import com.team2898.engine.types.Timestamp
import com.team2898.robot.config.NavxConf.*
import com.team2898.engine.logic.ILooper
import com.team2898.engine.logic.ISelfCheck
import com.team2898.engine.logic.GamePeriods
import com.team2898.engine.logic.RunEvery


object Navx: ISelfCheck, ILooper {

    override val enableTimes = listOf(GamePeriods.AUTO, GamePeriods.TELEOP)

    val navx = AHRS(NAVX_PORT)

    var angleAdjustment = Rotation2d()


    // in degrees
    val yaw: Double
        get() = navx.yaw.toDouble()
    // in degrees/second
    val yawRate: Double
        get() = navx.rate.toDouble()
    val rotation: Rotation2d
        get() = Rotation2d.createFromDegrees(yaw)

    val every = RunEvery(LOG_EVERY)
    override val loop = AsyncLooper(UPDATE_HZ) {

        if (every.shouldRun()) {
            Logger.logData("NavX", "IMU_YAW", yaw)
            Logger.logData("NavX", "IMU_YAW_RATE", yawRate)
        }
    }

    @Synchronized fun reset() {
        navx.reset()
    }

    override fun selfCheckup(): Boolean {
        return false
    }

    override fun selfTest(): Boolean = selfCheckup()
}