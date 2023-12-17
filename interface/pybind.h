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
#include "core/data/map_database.h"
#include "core/config.h"

#include "scene_mvs.h"

#include "pcl_serializer.h"

#ifdef VISUAL_DEBUG
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
  // constructor from config file
  Config(const std::string &config_file_path);

  // vocabulary path
  Config& Vocab(const std::string& vocab_file_path);
  std::string vocab_file_path_ = "";

  // set camera model  
  Config& Model(const int imwidth, const int imheight);

  // enable line
  Config& LineTrack(bool flag = true);
  bool line_track_ = false;

  // mapping module
  Config& Mapping(bool flag = true);
  bool mapping_ = true;

  // load map database
  Config& Database(const std::string& map);
  bool preload = false;
  std::string map_db_path_ = "";

  // serialize for scene generation
  Config& Serialize(const std::string& map, const std::string& scene, const std::string& raw_img);
  bool serialize_ = false;
  std::string map_path_ = "";
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
  Session(const Config& cfg);

public:
  // add new image frame
  void addTrack(py::array_t<uint8_t>& input, double time_ms = -1);
  // get feature points
  py::array_t<float> getFeaturePoints();
  // get tracking status
  py::array_t<size_t> getTrackingState();
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
  cv::Mat getImageBGR(py::array_t<uint8_t>& input);
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

#ifdef VISUAL_DEBUG
  std::unique_ptr<pangolin_viewer::viewer> pviewer_ = nullptr;
  std::thread viewer_thread_;
#endif

};


PYBIND11_MODULE(pysfm, m) {
  m.doc() = "PLSLAM system wrapper modified from ORB-SLAM2 ";

  py::class_<Config>(m, "Config")
    .def(py::init<std::string>(), py::arg("config_file_path"))
    .def("vocab", &Config::Vocab, py::arg("path"))
    .def("model", &Config::Model, py::arg("imwidth"), py::arg("imheight"))
    .def("line_track", &Config::LineTrack, py::arg("flag") = true)
    .def("mapping", &Config::Mapping, py::arg("flag") = true)
    .def("database", &Config::Database, py::arg("map"))
    .def("serialize", &Config::Serialize, py::arg("map"), py::arg("scene"), py::arg("raw_img"));

  py::class_<Session>(m, "Session")
    .def(py::init<Config>(), py::arg("config"))
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("get_feature_points", &Session::getFeaturePoints)
    .def("get_position_cv", &Session::getTwc)
    .def("get_position_gl", &Session::getTwcGL)
    .def("get_position_three", &Session::getTwcThree)
    .def("get_map_protobuf", &Session::getMapProtoBuf)
    .def("release", &Session::release)
    .def("cancel", &Session::cancel);
}

#endif // __PYBIND_H__