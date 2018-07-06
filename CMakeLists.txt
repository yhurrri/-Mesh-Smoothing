cmake_minimum_required(VERSION 2.8.12)
project(000_cw2_smooth)



add_executable(${PROJECT_NAME}_bin main.cpp normalEstimation.cpp normalEstimation.h normalEstimation.hpp cloudManager.cpp cloudManager.h cloudManager.hpp decoratedCloud.cpp decoratedCloud.h decoratedCloud.hpp typedefs.h
        ANNsrc/bd_tree.h
        ANNsrc/kd_tree.h
        ANNsrc/kd_split.h
        ANNsrc/kd_search.h
        ANNsrc/kd_pr_search.h
        ANNsrc/kd_util.h
        ANNsrc/kd_fix_rad_search.h
        ANNsrc/pr_queue.h
        ANNsrc/pr_queue_k.h
        ANNsrc/ANN.cpp
        ANNsrc/bd_fix_rad_search.cpp
        ANNsrc/bd_pr_search.cpp
        ANNsrc/bd_search.cpp
        ANNsrc/bd_tree.cpp
        ANNsrc/brute.cpp
        ANNsrc/kd_dump.cpp
        ANNsrc/kd_tree.cpp
        ANNsrc/kd_split.cpp
        ANNsrc/kd_search.cpp
        ANNsrc/kd_pr_search.cpp
        ANNsrc/kd_util.cpp
        ANNsrc/kd_fix_rad_search.cpp
        ANNsrc/perf.cpp Function.cpp Function.h
        spectra
        )
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::viewer tutorials)
