package com.team2898.engine.extensions.Vector2D

import com.team2898.engine.math.linear.rotateVector2D
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import java.util.*

//**************************************************
// NOTE: VECTOR2D IS IMMUTABLE --- THESE EXTENSIONS RETURN, NOT MUTATE
//**************************************************

fun Vector2D.rotate(other: Vector2D): Vector2D = rotateVector2D(this, other)

operator fun Vector2D.minus(other: Vector2D): Vector2D = minus(other)

operator fun Vector2D.plus(other: Vector2D): Vector2D = add(other)

operator fun Vector2D.times(scalar: Double): Vector2D = scalarMultiply(scalar)

operator fun Vector2D.div(scalar: Double): Vector2D = scalarMultiply(1 / scalar)

val Vector2D.l1: Double
    get() = norm1

val Vector2D.l2: Double
    get() = norm

val Vector2D.atan2: Double
    get() = Math.atan2(this.y, this.x)
