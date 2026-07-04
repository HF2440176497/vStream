

#include "base.hpp"

// Note: do NOT call google::InitGoogleLogging() here.
// The framework's GlogLevelInitializer (framework/core/src/cnstream_logging.cpp)
// has already invoked it during static initialization. A second call would
// trip glog's CHECK(!IsGoogleLoggingInitialized()) and abort the process.

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    google::ShutdownGoogleLogging();
    return ret;
}
