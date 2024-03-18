#ifndef __FORMAT_COLMAP_H__
#define __FORMAT_COLMAP_H__

#include <unordered_map>
#include <memory>
#include <fstream>
#include <filesystem>
#include <memory>

#include "core/system.h"
#include "core/data/keyframe.h"
#include "core/data/landmark.h"

namespace Colmap{

class Sparse{
friend class PLSLAM::system;
public:
    // constructor
    Sparse(const std::shared_ptr<PLSLAM::system>&);
    // serialize
    void serialize(const std::string& work_dir);

protected:
    // filter keyframes
    void filterKeyframes();

    // filter landmarks
    void filterLandmarks();

    // serialize camera
    void serializeCamera();

    // serialize images
    void serializeImages();

    // serialize mappoints
    void serializeMapPoints();

    // filtered keyframes
    std::unordered_map<unsigned long, PLSLAM::data::keyframe*> keyframes_;

    // filtered landmarks
    std::unordered_map<unsigned int, PLSLAM::data::landmark*> landmarks_;

    // Slam PLSLAM system
    std::shared_ptr<PLSLAM::system> psystem_ = nullptr;

    // workdir
    std::string work_dir_;
    std::string image_dir_;
    std::string sparse_dir_;

    // coordinate convert
    bool convert_cv2gl_ = true;

};


} // namespace Colmap

#endif
