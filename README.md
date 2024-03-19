# PySLAM

### Based on [Structure-PLP-SLAM](https://github.com/PeterFWS/Structure-PLP-SLAM)

### Modified

* Remove the Plan-Segment part, use points and lines only.
* Use fbow to replace DBoW2.
* Fix some bug about freezing and segment fault.

### Interface
* Add python binding using [pybind11](https://github.com/pybind/pybind11)(currently monocular only, modify on your own to support more types of camera).
* Integrate [COLMAP](https://github.com/colmap/colmap) txet format as additional output (no databse.db). 
* Integrate generating [OpenMVS](https://github.com/cdcseacave/openMVS) scene through [official interface](https://github.com/cdcseacave/openMVS/blob/master/libs/MVS/Interface.h).

### Build
* Make sure you have installed `opencv 3.4.16+` or `opencv 4`
* `pangolin`, `openMVS` and `socket.io` are optional.
* Change the CMakeLists.txt on your own purpose.
```bash
# build thrid party
./build_3rd.sh

# build slam
mkdir build
cd build
cmake .. -DBUILD_PYBIND=ON -DBUILD_PANGOLIN_VIEWER=ON -DBUILD_EXAMPLES=ON -DUSE_PANGOLIN_VIEWER=ON
make -j
```
