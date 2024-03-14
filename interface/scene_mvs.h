#ifndef __SCENE_MVS_H__
#define __SCENE_MVS_H__

#include <unordered_map>
#include <memory>

#include "interface_mvs.h"
#include "core/system.h"
#include "core/data/keyframe.h"

namespace MVS {

class Scene {
friend class PLSLAM::system;
public:
  // constructor
  Scene(const std::shared_ptr<PLSLAM::system>&);
  // serialize
  void serialize(const std::string& filename, const std::string& image_dir);

protected:
  // filter keyframes
  void filterKeyframes();

  // check definition
  uint32_t getMaxDefinition(const std::vector<PLSLAM::data::keyframe*>& keyframes, uint32_t start = 0, uint32_t end = -1);

  // define platforms with camera intrinsic
  void definePlatform();

  // define image and pose
  void defineImagePose(const std::string& image_dir);

  // define point cloud
  void defineStructure();

  // map keyframe id
  void bindKeyframeID();
  uint32_t getBindedID(unsigned long kfid);

  // filtered keyframes
  std::vector<PLSLAM::data::keyframe*> keyframes_;

  // keyframe id maps
  std::unordered_map<unsigned long, uint32_t> kfid_map_;

  // openMVS interface
  _INTERFACE_NAMESPACE::Interface scene_;

  // Slam PLSLAM system
  std::shared_ptr<PLSLAM::system> psystem_ = nullptr;

  // coordinate convert
  bool convert_cv2gl_ = true;

};



}; // namespace MVS

#endif //  __SCENE_MVS_H__