package com.team2898.robot.config.DrivetrainConf

import com.team2898.robot.subsystems.Drivetrain

const val KP_HIGH = 0.0
const val KI_HIGH = 0.0
const val KD_HIGH = 0.0
const val KF_HIGH = 0.0

const val KP_LOW = 0.1
const val KI_LOW = 0.001
const val KD_LOW = 0.0
const val KF_LOW = 1023/9493.94

const val MAX_SPEED_HIGH = 0.0
const val MAX_SPEED_LOW = 0.0

const val HIGH_GEAR_RAMPRATE = 0.0
const val LOW_GEAR_RAMPRATE = 0.0

const val OPENLOOP_ACCELERATE = 1.0
const val OPENLOOP_DECELERATE = 3.0

const val SOMETHING_ACCELERATE = 1.0
const val SOMETHING_DECELERATE = 3.0

const val LOW_GEAR_MAX_AMPS = 40
const val HIGH_GEAR_MAX_AMPS = 20

val DIRECT_CONTROL_METHOD = Drivetrain.ControlModes.OPEN_LOOP

const val LEFT_MM_STRAIGHT_ADJ_FACTOR = 1.0
