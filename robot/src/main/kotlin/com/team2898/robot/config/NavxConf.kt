package com.team2898.robot.config.NavxConf

import edu.wpi.first.wpilibj.SPI

val NAVX_PORT = SPI.Port.kMXP
val AVG_PERIOD_MS = 25
val UPDATE_HZ = 100.0
val DATA_STORAGE_TIME_MS = 250
val LOG_HZ = 10.0














// Ancillary
val FILTER_BUFFER_SIZE = Math.round(AVG_PERIOD_MS.toDouble()/(1000.0/UPDATE_HZ)).toInt()
val BUFFER_SIZE = Math.round((DATA_STORAGE_TIME_MS.toDouble())/(1000.0/UPDATE_HZ)).toInt()
val LOG_EVERY = Math.round(UPDATE_HZ/LOG_HZ).toInt()
