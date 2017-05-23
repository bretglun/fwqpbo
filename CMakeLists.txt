project (fw)

set (eigen_path ~/Kod/recon_ks/include/eigen/)

include_directories(cpp cpp/QPBO-v1.4.src ${eigen_path})

add_library (fw SHARED cpp/FW.cpp cpp/fibonacci.cpp cpp/QPBO-v1.4.src/QPBO.cpp cpp/QPBO-v1.4.src/QPBO_extra.cpp cpp/QPBO-v1.4.src/QPBO_maxflow.cpp cpp/QPBO-v1.4.src/QPBO_postprocessing.cpp)
