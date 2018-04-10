package com.team2898.robot.motion

import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.logic.GamePeriods
import com.team2898.engine.logic.ILooper

object RobotState: ILooper {
    override val enableTimes = listOf(GamePeriods.TELEOP, GamePeriods.AUTO)
    override val loop = AsyncLooper(100.0) {

    }
}