package com.team2898.engine.kinematics

import com.team2898.engine.math.linear.rotateVector2D
import com.team2898.engine.types.Interpolable
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import java.text.DecimalFormat


// Represents a 2D translation on the x, y plane
class Translation2d: Interpolable<Translation2d> {
    private var position = Vector2D(0.0, 0.0)

    var x: Double
        get() {
            return position.x
        }
        set(value) {position = Vector2D(value, y)}

    var y: Double
        get() {
            return position.y
        }
        set(value) {position = Vector2D(x, value)}


    constructor() {
    }

    constructor(x: Double, y: Double) {
        position = Vector2D(x, y)
    }

    constructor(toCopy: Translation2d) {
        position = Vector2D(toCopy.x, toCopy.y)
    }

    constructor(vector: Vector2D) {
        position = vector
    }

    /**
    * Normalized distance (distance to origin)
    */
    fun norm(): Double = position.distance(Vector2D.ZERO)

    fun normalized(): Vector2D = position.normalize()

    fun vector(): Vector2D = position

    fun translateBy(other: Translation2d): Translation2d {
        return Translation2d(position.x + other.x, position.y + other.y)
    }

    fun translateBy(other: Vector2D): Translation2d {
        return Translation2d(position.x + other.x, position.y + other.y)
    }

    fun rotateByOrigin(rotation: Rotation2d): Translation2d {
        return Translation2d(rotateVector2D(position, rotation.rotation))
    }

    fun inverse(): Translation2d {
        return Translation2d(-position.x, -position.y)
    }

    override fun interpolate(upperVal: Translation2d, interpolatePoint: Double): Translation2d {
        when {
            (interpolatePoint <= 0) -> return Translation2d(this)
            (interpolatePoint >= 1) -> return Translation2d(upperVal)
            else -> return extrapolate(upperVal, interpolatePoint)
        }
    }

    fun extrapolate(slopePoint: Translation2d, extrapolatePoint: Double): Translation2d {
        return Translation2d(
                extrapolatePoint * (slopePoint.x - position.x) + position.x,
                extrapolatePoint * (slopePoint.y - position.y) + position.y
        )
    }

    override fun toString(): String {
        return "[${"%.3f".format(position.x)}, ${"%.3f".format(position.y)}]"
    }
}
