#include "scene_mvs.h"

#include "core/camera/perspective.h"

#include "core/data/map_database.h"
#include "core/data/keyframe.h"
#include "core/data/landmark.h"

#include <opencv2/core/eigen.hpp>

#include <spdlog/spdlog.h>

namespace MVS {

const Eigen::Matrix3d MAT_X33D_CV2GL_ = 
  (Eigen::Matrix3d() << 
    1, 0, 0,
    0, -1, 0,
    0, 0, -1).finished();

const Eigen::Matrix3d MAT_X33D_CV2GL_INVERSE_
   = MAT_X33D_CV2GL_;

const unsigned int KEYFRAME_LIMIT = 800;


Scene::Scene(const std::shared_ptr<PLSLAM::system>& psystem) 
  : psystem_(psystem)
{
  // initialize  
}

void Scene::serialize(const std::string& work_dir) {
  std::string base_dir = work_dir;
  if(base_dir.back() != '/')  base_dir += "/";
  std::filesystem::create_directories(base_dir + "scene/");

  // set convert flag
  convert_cv2gl_ = true;
  // filter keyframes
  spdlog::info("filtering keyframes");
  filterKeyframes();
  // bind keyframe id
  spdlog::info("binding keyframe id");
  bindKeyframeID();
  // set platform
  spdlog::info("defining platform");
  definePlatform();
  // set image and pose
  spdlog::info("defining image and pose");
  defineImagePose(base_dir + "images/");
  // set map point
  spdlog::info("defining structure"); 
  defineStructure();

  // serialize
  std::string filepath = base_dir + "scene/scene.mvs";
  bool res = _INTERFACE_NAMESPACE::ARCHIVE::SerializeSave(scene_, filepath);

  if(res) spdlog::info("mvs scene serialized to {}", filepath);
  else    spdlog::error("mvs scene serialization failed");
}

uint32_t Scene::getMaxDefinition(const std::vector<PLSLAM::data::keyframe*>& keyframes, uint32_t start, uint32_t end) {
  if(end == -1 || end > keyframes.size()) {
    end = keyframes.size();
  }

  uint32_t argmax = start;
  double max_mean_value = 0;

  for(auto i = start; i < end; i++) {
    auto& kf = keyframes[i];
    if(!kf) continue;
    cv::Mat img = kf->get_img_rgb();
    if(img.empty()) continue;
    cv::Mat gray, laplacian;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    double mean_value = cv::mean(laplacian)[0];

    if(mean_value > max_mean_value) {
      max_mean_value = mean_value;
      argmax = i;
    }
  }

  return argmax;
}


void Scene::filterKeyframes() {
  std::vector<PLSLAM::data::keyframe*> keyframes_origin, keyframes_filter;
  keyframes_origin = psystem_->map_db_->get_all_keyframes();
  // sort by time
  std::sort(keyframes_origin.begin(), keyframes_origin.end(), 
    [](const PLSLAM::data::keyframe* a, const PLSLAM::data::keyframe* b) {
      return a->id_ < b->id_;
    });
  // filter
  for(auto& kf : keyframes_origin) {
    // check status
    if(!kf || kf->will_be_erased()) continue;
    // check image
    if(kf->get_img_rgb().empty()) continue;
    
    keyframes_filter.push_back(kf);
  }

  // downsample
  auto kf_size = keyframes_filter.size();
  // int step = kf_size / KEYFRAME_LIMIT;
  // step = (step == 0 ? 1 : step);
  int step = 1;
  for(int i = 0; i < kf_size; i += step) {
    // find max definition from each step
    // const uint32_t arg = getMaxDefinition(keyframes_filter, i, i + step);
    keyframes_.push_back(keyframes_filter[i]);
  }
}

void Scene::bindKeyframeID() {
  // mapping
  uint32_t cnt = 0;
  for(auto& kf : keyframes_) {
    kfid_map_[kf->id_] = cnt++;
  }
}

uint32_t Scene::getBindedID(unsigned long kfid) {
  if (kfid_map_.find(kfid) == kfid_map_.end()) 
    return (uint32_t)-1;
  return kfid_map_[kfid];
}

void Scene::definePlatform() {
  // platform
  _INTERFACE_NAMESPACE::Interface::Platform platform;
  // camera
  _INTERFACE_NAMESPACE::Interface::Platform::Camera camera;
  
  // origin data
  cv::Mat K = dynamic_cast<PLSLAM::camera::perspective*>(psystem_->camera_)->cv_cam_matrix_;
  K.convertTo(K, CV_64F);

  // set camera
  camera.width = psystem_->camera_->cols_;
  camera.height = psystem_->camera_->rows_;
  camera.K = K;

  // sub-pose
  camera.R = cv::Matx33d::eye();
  camera.C = cv::Point3d(0, 0, 0);
  platform.cameras.emplace_back(camera);

  // push to scene
  scene_.platforms.emplace_back(platform);

  spdlog::info("MVS: platform defined");
}

void Scene::defineImagePose(const std::string& image_dir) {
  auto& platform = scene_.platforms[0];

  size_t n_views = keyframes_.size();
  
  scene_.images.reserve(n_views);
  platform.poses.reserve(n_views);

  size_t cnt = 0;

  for(auto& kf : keyframes_) {
    _INTERFACE_NAMESPACE::Interface::Image image;
    // image source
    image.ID = getBindedID(kf->id_);
    image.name = "../../images/" + std::to_string(kf->id_) + ".png";
    
    cv::Mat img = kf->get_img_rgb();
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    const std::string filepath = image_dir + std::to_string(kf->id_) + ".png";
    cv::imwrite(filepath, img);

    // camera
    image.platformID = 0;
    image.cameraID = 0;
    // pose
    _INTERFACE_NAMESPACE::Interface::Platform::Pose pose;
    image.poseID = platform.poses.size();

    // rotation
    Eigen::Matrix3d rcw = kf->get_rotation();
    if(convert_cv2gl_) 
      rcw = rcw * MAT_X33D_CV2GL_INVERSE_;
    cv::eigen2cv(rcw, pose.R);

    // center (translation)
    auto t = kf->get_cam_center();
    if(convert_cv2gl_) 
      pose.C = cv::Point3d(t(0), -t(1), -t(2));
    else
      pose.C = cv::Point3d(t(0), t(1), t(2));

    platform.poses.push_back(pose);
    scene_.images.emplace_back(image);
    ++cnt;
  }

  spdlog::info("MVS: {} images and poses defined", cnt);
}

void Scene::defineStructure() {
  auto map_points = psystem_->map_db_->get_all_landmarks();
  size_t cnt = 0;

  for(auto& mp : map_points ) {
    if(!mp || mp->will_be_erased())
      continue;

    _INTERFACE_NAMESPACE::Interface::Vertex vert;
    auto& views = vert.views;

    // set observation
    auto observation = mp->get_observations();
    for(auto& obs : observation) {
      
      uint32_t img_id = getBindedID(obs.first->id_);
      if(img_id == -1)
        continue;

      _INTERFACE_NAMESPACE::Interface::Vertex::View view;
      view.imageID = img_id;
      view.confidence = 0;
      views.emplace_back(view);
    }
    // sort image
    if(views.size() < 2)
      continue;

    std::sort(views.begin(), views.end(), 
      [](const _INTERFACE_NAMESPACE::Interface::Vertex::View& a, 
          const _INTERFACE_NAMESPACE::Interface::Vertex::View& b) {
        return a.imageID < b.imageID;
      });

    // set 3D position
    auto p = mp->get_pos_in_world();
    if(convert_cv2gl_) 
      vert.X = cv::Point3f(p(0), -p(1), -p(2));
    else
      vert.X = cv::Point3f(p(0), p(1), p(2));
    
    scene_.vertices.emplace_back(vert);
    ++cnt;
  }

  spdlog::info("MVS: {} points defined", cnt);
}

}; // namespace MVS