#include "shit/elbow.cc"

#include <vector>

#include "frc971/control_loops/state_feedback_loop.h"

namespace shit {

StateFeedbackPlantCoefficients<2, 1, 1> MakeElbowPlantCoefficients() {
  Eigen::Matrix<double, 1, 2> C;
  C(0, 0) = 1;
  C(0, 1) = 0;
  Eigen::Matrix<double, 1, 1> D;
  D(0, 0) = 0;
  Eigen::Matrix<double, 1, 1> U_max;
  U_max(0, 0) = 12.0;
  Eigen::Matrix<double, 1, 1> U_min;
  U_min(0, 0) = -12.0;
  Eigen::Matrix<double, 2, 2> A;
  A(0, 0) = 1.0;
  A(0, 1) = 0.004988388563313852;
  A(1, 0) = 0.0;
  A(1, 1) = 0.99535902345735339;
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) = 1.0;
  A_inv(0, 1) = -0.0050116475018097649;
  A_inv(1, 0) = 0.0;
  A_inv(1, 1) = 1.0046626156324241;
  Eigen::Matrix<double, 2, 1> B;
  B(0, 0) = 4.3074960902506945e-06;
  B(1, 0) = 0.0017216636367007705;
  return StateFeedbackPlantCoefficients<2, 1, 1>(A, A_inv, B, C, D, U_max, U_min);
}

StateFeedbackControllerCoefficients<2, 1, 1> MakeElbowControllerCoefficients() {
  Eigen::Matrix<double, 1, 2> K;
  K(0, 0) = 59.166949214182701;
  K(0, 1) = 16.090296239800114;
  Eigen::Matrix<double, 1, 2> Kff;
  Kff(0, 0) = 0.00051734057104117705;
  Kff(0, 1) = 580.83354753769481;
  return StateFeedbackControllerCoefficients<2, 1, 1>(K, Kff);
}

StateFeedbackObserverCoefficients<2, 1, 1> MakeElbowObserverCoefficients() {
  Eigen::Matrix<double, 2, 1> L;
  L(0, 0) = 1.0556488825038899;
  L(1, 0) = 37.785381367517857;
  return StateFeedbackObserverCoefficients<2, 1, 1>(L);
}

StateFeedbackPlant<2, 1, 1> MakeElbowPlant() {
  ::std::vector< ::std::unique_ptr<StateFeedbackPlantCoefficients<2, 1, 1>>> plants(1);
  plants[0] = ::std::unique_ptr<StateFeedbackPlantCoefficients<2, 1, 1>>(new StateFeedbackPlantCoefficients<2, 1, 1>(MakeElbowPlantCoefficients()));
  return StateFeedbackPlant<2, 1, 1>(&plants);
}

StateFeedbackController<2, 1, 1> MakeElbowController() {
  ::std::vector< ::std::unique_ptr<StateFeedbackControllerCoefficients<2, 1, 1>>> controllers(1);
  controllers[0] = ::std::unique_ptr<StateFeedbackControllerCoefficients<2, 1, 1>>(new StateFeedbackControllerCoefficients<2, 1, 1>(MakeElbowControllerCoefficients()));
  return StateFeedbackController<2, 1, 1>(&controllers);
}

StateFeedbackObserver<2, 1, 1> MakeElbowObserver() {
  ::std::vector< ::std::unique_ptr<StateFeedbackObserverCoefficients<2, 1, 1>>> observers(1);
  observers[0] = ::std::unique_ptr<StateFeedbackObserverCoefficients<2, 1, 1>>(new StateFeedbackObserverCoefficients<2, 1, 1>(MakeElbowObserverCoefficients()));
  return StateFeedbackObserver<2, 1, 1>(&observers);
}

StateFeedbackLoop<2, 1, 1, StateFeedbackPlant<2, 1, 1>, StateFeedbackObserver<2, 1, 1>> MakeElbowLoop() {
  return StateFeedbackLoop<2, 1, 1, StateFeedbackPlant<2, 1, 1>, StateFeedbackObserver<2, 1, 1>>(MakeElbowPlant(), MakeElbowController(), MakeElbowObserver());
}

}  // namespace shit
