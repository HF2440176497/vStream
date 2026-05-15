
#include "base.hpp"
#include <opencv2/core/utility.hpp>

int main(int argc, char **argv) {
    cv::setNumThreads(0);
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    google::ShutdownGoogleLogging();
    return ret;
}
