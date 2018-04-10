package com.team2898.robot.config.WristConf

import com.team2898.engine.kinematics.Rotation2d

const val MAX_ACCEL: Double = 0.25
const val MAX_DEACCEL: Double = 0.25

const val Kp: Double = 2.0
const val Ki: Double = 0.0
const val Kd: Double = 7.5

const val MAX_FORWARD_POSITION = 1.0
const val MAX_REVERSE_POSITION = -1.0


const val LOOP_HZ: Double = 100.0

const val ABSOLUTE_OFFSET = 1861

const val motion_magic_acceleration = 60.0
const val motion_magic_cruise_velocity = 60.0
