#ifndef SHIT_ELBOWI_CC_
#define SHIT_ELBOWI_CC_

#include "frc971/control_loops/state_feedback_loop.h"

namespace shit {

StateFeedbackPlantCoefficients<3, 1, 1> MakeIntegralElbowPlantCoefficients();

StateFeedbackControllerCoefficients<3, 1, 1> MakeIntegralElbowControllerCoefficients();

StateFeedbackObserverCoefficients<3, 1, 1> MakeIntegralElbowObserverCoefficients();

StateFeedbackPlant<3, 1, 1> MakeIntegralElbowPlant();

StateFeedbackController<3, 1, 1> MakeIntegralElbowController();

StateFeedbackObserver<3, 1, 1> MakeIntegralElbowObserver();

StateFeedbackLoop<3, 1, 1, StateFeedbackPlant<3, 1, 1>, StateFeedbackObserver<3, 1, 1>> MakeIntegralElbowLoop();

}  // namespace shit

#endif  // SHIT_ELBOWI_CC_
