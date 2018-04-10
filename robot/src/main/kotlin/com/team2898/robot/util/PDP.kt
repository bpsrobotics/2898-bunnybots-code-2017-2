package com.team2898.robot.util

import com.team2898.robot.config.RobotMap.*
import edu.wpi.first.wpilibj.PowerDistributionPanel

import java.util.HashMap

object PDP {

    val pdp = PowerDistributionPanel(PDP_CANID) // TODO

    /** volts
     */
    val batteryVoltage: Double
        get() = pdp.voltage

    /** watts
     */
    val usedPower: Double
        get() = pdp.totalPower

    /** amps
     */
    val totalCurrent: Double
        get() = pdp.totalCurrent

    /** joules
     */
    val usedEnergy: Double
        get() = pdp.totalEnergy

    /** resets energy and power count
     */
    fun resetUsedEnergy() {
        pdp.resetTotalEnergy()
    }

    /** gets the current from a single current
     */
    fun getChannelCurrent(channel: Int): Double {
        return pdp.getCurrent(channel)
    }
}
