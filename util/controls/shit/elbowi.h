#include "shit/elbowi.cc"

#include <vector>

#include "frc971/control_loops/state_feedback_loop.h"

namespace shit {

StateFeedbackPlantCoefficients<3, 1, 1> MakeIntegralElbowPlantCoefficients() {
  Eigen::Matrix<double, 1, 3> C;
  C(0, 0) = 1.0;
  C(0, 1) = 0.0;
  C(0, 2) = 0.0;
  Eigen::Matrix<double, 1, 1> D;
  D(0, 0) = 0;
  Eigen::Matrix<double, 1, 1> U_max;
  U_max(0, 0) = 12.0;
  Eigen::Matrix<double, 1, 1> U_min;
  U_min(0, 0) = -12.0;
  Eigen::Matrix<double, 3, 3> A;
  A(0, 0) = 1.0;
  A(0, 1) = 0.004988388563313852;
  A(0, 2) = 4.3074960902506945e-06;
  A(1, 0) = 0.0;
  A(1, 1) = 0.99535902345735339;
  A(1, 2) = 0.0017216636367007705;
  A(2, 0) = 0.0;
  A(2, 1) = 0.0;
  A(2, 2) = 1.0;
  Eigen::Matrix<double, 3, 3> A_inv;
  A_inv(0, 0) = 1.0;
  A_inv(0, 1) = -0.0050116475018097649;
  A_inv(0, 2) = 4.3208751735774368e-06;
  A_inv(1, 0) = 0.0;
  A_inv(1, 1) = 1.0046626156324241;
  A_inv(1, 2) = -0.0017296910924870276;
  A_inv(2, 0) = 0.0;
  A_inv(2, 1) = 0.0;
  A_inv(2, 2) = 1.0;
  Eigen::Matrix<double, 3, 1> B;
  B(0, 0) = 4.3074960902506945e-06;
  B(1, 0) = 0.0017216636367007705;
  B(2, 0) = 0.0;
  return StateFeedbackPlantCoefficients<3, 1, 1>(A, A_inv, B, C, D, U_max, U_min);
}

StateFeedbackControllerCoefficients<3, 1, 1> MakeIntegralElbowControllerCoefficients() {
  Eigen::Matrix<double, 1, 3> K;
  K(0, 0) = 59.166949214182701;
  K(0, 1) = 16.090296239800114;
  K(0, 2) = 1.0;
  Eigen::Matrix<double, 1, 3> Kff;
  Kff(0, 0) = 0.00051734057104117705;
  Kff(0, 1) = 580.83354753769481;
  Kff(0, 2) = 0.0;
  return StateFeedbackControllerCoefficients<3, 1, 1>(K, Kff);
}

StateFeedbackObserverCoefficients<3, 1, 1> MakeIntegralElbowObserverCoefficients() {
  Eigen::Matrix<double, 3, 1> L;
  L(0, 0) = 0.98452287463577381;
  L(1, 0) = 33.443502534667033;
  L(2, 0) = 12.834556607058122;
  return StateFeedbackObserverCoefficients<3, 1, 1>(L);
}

StateFeedbackPlant<3, 1, 1> MakeIntegralElbowPlant() {
  ::std::vector< ::std::unique_ptr<StateFeedbackPlantCoefficients<3, 1, 1>>> plants(1);
  plants[0] = ::std::unique_ptr<StateFeedbackPlantCoefficients<3, 1, 1>>(new StateFeedbackPlantCoefficients<3, 1, 1>(MakeIntegralElbowPlantCoefficients()));
  return StateFeedbackPlant<3, 1, 1>(&plants);
}

StateFeedbackController<3, 1, 1> MakeIntegralElbowController() {
  ::std::vector< ::std::unique_ptr<StateFeedbackControllerCoefficients<3, 1, 1>>> controllers(1);
  controllers[0] = ::std::unique_ptr<StateFeedbackControllerCoefficients<3, 1, 1>>(new StateFeedbackControllerCoefficients<3, 1, 1>(MakeIntegralElbowControllerCoefficients()));
  return StateFeedbackController<3, 1, 1>(&controllers);
}

StateFeedbackObserver<3, 1, 1> MakeIntegralElbowObserver() {
  ::std::vector< ::std::unique_ptr<StateFeedbackObserverCoefficients<3, 1, 1>>> observers(1);
  observers[0] = ::std::unique_ptr<StateFeedbackObserverCoefficients<3, 1, 1>>(new StateFeedbackObserverCoefficients<3, 1, 1>(MakeIntegralElbowObserverCoefficients()));
  return StateFeedbackObserver<3, 1, 1>(&observers);
}

StateFeedbackLoop<3, 1, 1, StateFeedbackPlant<3, 1, 1>, StateFeedbackObserver<3, 1, 1>> MakeIntegralElbowLoop() {
  return StateFeedbackLoop<3, 1, 1, StateFeedbackPlant<3, 1, 1>, StateFeedbackObserver<3, 1, 1>>(MakeIntegralElbowPlant(), MakeIntegralElbowController(), MakeIntegralElbowObserver());
}

}  // namespace shit
