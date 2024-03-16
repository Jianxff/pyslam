#ifndef __PYBIND_H__
#define __PYBIND_H__

#include <thread>
#include <memory>
#include <queue>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <spdlog/spdlog.h>

#include <yaml-cpp/yaml.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "core/system.h"
#include "core/tracking_module.h"
#include "core/publish/frame_publisher.h"
#include "core/data/map_database.h"
#include "core/config.h"

#include "scene_mvs.h"

#include "pcl_serializer.h"

#ifdef WITH_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

namespace py = pybind11;

class ImageStream{
public:
  ImageStream(bool force_realtime = false);
  // add new image to image queue
  void addNewImage(cv::Mat& im, double time_ms);
  // get new image from image queue
  bool getNewImage(cv::Mat& im, double& time);
  // release stream
  void release();

private:
  bool force_realtime_ = false;
  bool new_img_abailable_ = false;
  bool released_ = false;
  std::mutex img_mutex_;
  std::queue<cv::Mat> img_stream_;
  std::queue<double> img_times_;

};


class Config {
public:
  Config(const std::string& config_file_path,
          const std::string & vocab_file_path,
          const bool mapping = true,
          const bool line_track = false,
          const bool loop_detect = false,
          const bool viewer = false,
          const bool loadmap = false,
          const bool serialize = false,
          const std::string& map_db = "",
          const std::string& scene_path = "",
          const std::string& raw_img_dir = "");

  // reset camera model  
  Config& fitmodel(const int imwidth, const int imheight);

  // vocab file
  std::string vocab_file_path_ = "";
  // enable line
  bool line_track_ = false;
  // mapping module
  bool mapping_ = true;
  // enable loopdetect
  bool loop_detect_ = false;
  // load map database
  bool preload = false;
  std::string map_db_path_ = "";
  // pangolin viewer
  bool viewer_ = false;
  // serialize for scene generation
  bool serialize_ = false;
  std::string scene_path_ = "";
  std::string raw_image_path_ = "";

  // instance origin config
  std::shared_ptr<PLSLAM::config> instance();

protected:
  std::shared_ptr<PLSLAM::config> pconfig_ = nullptr;
  YAML::Node yaml_node_;
  std::string config_file_path_;
};


class Session {
public:
  // constructor from config file
  Session(const Config& cfg, const bool sync = false);

public:
  // add new image frame
  void addTrack(py::array_t<uint8_t>& input, double time_ms = -1);
  // get feature points
  py::array_t<float> getFeaturePoints();
  // get tracking status
  py::array_t<size_t> getTrackingState();
  // get tracking visualize
  py::array_t<uint8_t> getTrackingVisualize();
  // get camera twc
  Eigen::Matrix4d getTwc();
  // get camera twc under openGL coordinate
  Eigen::Matrix4d getTwcGL();
  // get camear position on three
  Eigen::Matrix4d getTwcThree();
  // get serialize data
  std::string getMapProtoBuf();
  // stop session
  py::array_t<size_t> release();
  // cancel session
  void cancel();

private:
  // run thread
  void run();
  // get image from webrtc frame
  cv::Mat getImageRGB(py::array_t<uint8_t>& input);
  // dump image
  void dumpImages();

  std::atomic<bool> released_ = false;
  std::atomic<bool> exit_required_ = false;

  Eigen::Matrix4d Twc_ = Eigen::Matrix4d::Identity();

  Config cfg_;
  std::shared_ptr<PLSLAM::system> psystem_ = nullptr;
  std::shared_ptr<ImageStream> pstream_ = nullptr;
  std::shared_ptr<MVS::Scene> pmvs_ = nullptr;
  std::unique_ptr<pcl_serializer> pserializer_ = nullptr;

  std::thread system_thread_;

#ifdef WITH_PANGOLIN_VIEWER
  std::unique_ptr<pangolin_viewer::viewer> pviewer_ = nullptr;
  std::thread viewer_thread_;
#endif

};


PYBIND11_MODULE(pysfm, m) {
  m.doc() = "PLSLAM system wrapper modified from ORB-SLAM2 ";

  py::class_<Config>(m, "Config")
    .def(py::init<
        const std::string&, 
        const std::string&, 
        const bool, 
        const bool, 
        const bool, 
        const bool, 
        const bool, 
        const bool,
        const std::string&, 
        const std::string&, 
        const std::string&
      >(), 
      py::arg("config_file_path"), 
      py::arg("vocab_file_path"), 
      py::arg("mapping") = true, 
      py::arg("line_track") = false, 
      py::arg("loop_detect") = false, 
      py::arg("viewer") = false, 
      py::arg("loadmap") = false,
      py::arg("serialize") = false, 
      py::arg("map_db") = "", 
      py::arg("scene_path") = "", 
      py::arg("raw_img_dir") = ""
    )
    .def("fitmodel", &Config::fitmodel, py::arg("imwidth"), py::arg("imheight"));

  py::class_<Session>(m, "Session")
    .def(py::init<Config, bool>(), py::arg("config"), py::arg("sync") = false)
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("get_feature_points", &Session::getFeaturePoints)
    .def("get_position_cv", &Session::getTwc)
    .def("get_position_gl", &Session::getTwcGL)
    .def("get_position_three", &Session::getTwcThree)
    .def("get_tracking_visualize", &Session::getTrackingVisualize)
    .def("get_map_protobuf", &Session::getMapProtoBuf)
    .def("release", &Session::release)
    .def("cancel", &Session::cancel);
}

#endif // __PYBIND_H__