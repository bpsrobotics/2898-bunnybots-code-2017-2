package com.team2898.engine.controlLoops

import edu.wpi.first.wpilibj.util.BoundaryException
import edu.wpi.first.wpilibj.Timer

/** 
 * Standard PID control loop
 * Everything is synchronous, which means that you have to manually call the update() function
 * Adapted with modifications from Team 254's library
 * @author Solomon
 */

interface IPID {

     /** Take input, calculate output, and write to output. 
     * @param input
     *  input (sensor reading)
     */
     fun update(inputVal: Double, currentTime: Double = Timer.getFPGATimestamp()): Double

     /** Set PIDF gain
     * @param Kp
     * @param Ki
     * @param Kd
     */
     fun setPID(Kp: Double = 0.0, Ki: Double = 0.0, Kd: Double = 0.0, Kf: Double = 0.0) {}

     
     /** Sets whether or not the PID controller is continuous
     * @param continuous
     *  whether or not the controller input is continuous
     */
     fun setContinuous(continuous: Boolean = true) {}

     /** Sets PID deadband
     * This is the absolute range of values where proportional control will be turned off
     */
     fun setDeadband(deadband: Double) {}
     

     /** Sets max and min values that the input will be for the controller
     * 
     * @param minimumInput
     *  minimum input value accepted
     * @param maximumInput
     *  maximum input value accepted
     */
     fun setInputRange(minIn: Double, maxIn: Double) {}

     /** Sets min and max values that will be output
     * @param minimumOutput
     *  mininum value that will be output
     * @param maximumOutput
     *  maximum value that will be output
     */
     fun setOutputRange(minOut: Double, maxOut: Double) {}

     /** Sets setpoint for controller
     * @param setpoint
     *  setpoint desired
     */

     fun setSetpoint(setpoint: Double) {}

     /** Returns PID error
     * @return current error
     */
     fun getError(): Double {return 0.0}

     /** Is the error within error tolerance?
     * @return true if error < tolerance
     */
     fun onTarget(tolerance: Double): Boolean {return false}

     /**
     * Resets internal non-paramter values
     */
     fun reset() {}
 }
