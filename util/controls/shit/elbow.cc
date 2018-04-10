#ifndef SHIT_ELBOW_CC_
#define SHIT_ELBOW_CC_

#include "frc971/control_loops/state_feedback_loop.h"

namespace shit {

StateFeedbackPlantCoefficients<2, 1, 1> MakeElbowPlantCoefficients();

StateFeedbackControllerCoefficients<2, 1, 1> MakeElbowControllerCoefficients();

StateFeedbackObserverCoefficients<2, 1, 1> MakeElbowObserverCoefficients();

StateFeedbackPlant<2, 1, 1> MakeElbowPlant();

StateFeedbackController<2, 1, 1> MakeElbowController();

StateFeedbackObserver<2, 1, 1> MakeElbowObserver();

StateFeedbackLoop<2, 1, 1, StateFeedbackPlant<2, 1, 1>, StateFeedbackObserver<2, 1, 1>> MakeElbowLoop();

}  // namespace shit

#endif  // SHIT_ELBOW_CC_
