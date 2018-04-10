package com.team2898.engine.kinematics

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D

/** Stores a "twist" of our 2d position (2d linear + 2d angular component)
 * Modeled off of the ROS type
 * @param dx delta position, x component
 * @param dy delta position, y component
 * @param dtheta delta rotation, in radians
 */
class Twist2d {

    var dx: Double = 0.0
    var dy: Double = 0.0
    var dtheta: Double = 0.0

    constructor(dx: Double, dy: Double, dtheta: Double) {
        this.dx = dx
        this.dy = dy
        this.dtheta = dtheta
    }

    constructor(linear: Vector2D, angular: Vector2D) {
        this.dx = linear.x
        this.dy = linear.y
        this.dtheta = Math.atan2(angular.y, angular.x)
    }

    constructor (linear: Translation2d, angular: Rotation2d) {
        this.dx = linear.x
        this.dy = linear.y
        this.dtheta = angular.radians
    }
}
