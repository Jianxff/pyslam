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
#include "format_colmap.h"

#ifdef WITH_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

namespace py = pybind11;

class ImageStream{
public:
  ImageStream(bool sync = true);
  // add new image to image queue
  void addNewImage(cv::Mat& im, double time_ms);
  // get new image from image queue
  bool getNewImage(cv::Mat& im, double& time);
  // release stream
  void release();
  // status
  bool sync();
  size_t count();

private:
  bool sync_ = true;
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
        const bool savemap = false,
        const std::string& map_db = "",
        const bool convert_colmap = false,
        const std::string& colmap_dir = "",
        const bool convert_mvs = false,
        const std::string& mvs_dir = ""
    );

    // reset camera model  
    Config& fitmodel(const int imwidth, const int imheight);
    Config& setmodel(const int imwidth, const int imheight, const double fx, const double fy, const double k1 = 0.0, const double k2 = 0.0);

    // vocab file
    std::string vocab_file_path_ = "";
    // enable line
    bool linetrack_ = false;
    // mapping module
    bool mapping_ = true;
    // enable loopdetect
    bool loopclose_ = false;
    // load map database
    bool loadmap_ = false;
    bool savemap_ = false;
    std::string map_db_path_ = "";
    // pangolin viewer
    bool viewer_ = false;
    // serialize for scene generation
    bool convert_mvs_ = false;
    std::string mvs_dir_ = "";
    // serialize for colmap generation
    bool convert_colmap_ = false;
    std::string colmap_dir_ = "";

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
  // stop session
  py::array_t<size_t> release();
  // cancel session
  void cancel();

private:
  // run thread
  void run();
  // get image from webrtc frame
  cv::Mat getImageRGB(py::array_t<uint8_t>& input);

  std::atomic<bool> released_ = false;
  std::atomic<bool> exit_required_ = false;

  Eigen::Matrix4d Twc_ = Eigen::Matrix4d::Identity();

  Config cfg_;
  std::shared_ptr<PLSLAM::system> psystem_ = nullptr;
  std::shared_ptr<ImageStream> pstream_ = nullptr;
  std::shared_ptr<MVS::Scene> pmvs_ = nullptr;
  std::shared_ptr<Colmap::Sparse> pcolmap_ = nullptr;

  std::thread system_thread_;

#ifdef WITH_PANGOLIN_VIEWER
  std::unique_ptr<pangolin_viewer::viewer> pviewer_ = nullptr;
  std::thread viewer_thread_;
#endif

};


PYBIND11_MODULE(pyslam, m) {
  m.doc() = "python SLAM wrapper modified from Structure-PLP-SLAM";

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
        const bool,
        const std::string&, 
        const bool,
        const std::string&
      >(), 
      py::arg("config_file_path"), 
      py::arg("vocab_file_path"), 
      py::arg("mapping") = true, 
      py::arg("linetrack") = false, 
      py::arg("loopclose") = false, 
      py::arg("viewer") = false, 
      py::arg("loadmap") = false,
      py::arg("savemap") = false,
      py::arg("map_db_path") = "",
      py::arg("convert_colmap") = false,
      py::arg("colmap_dir") = "./",
      py::arg("convert_mvs") = false, 
      py::arg("mvs_dir") = "./"
    )
    .def("fitmodel", &Config::fitmodel, py::arg("imwidth"), py::arg("imheight"))
    .def("setmodel", &Config::setmodel, 
        py::arg("imwidth"), py::arg("imheight"), 
        py::arg("fx"), py::arg("fy"), 
        py::arg("k1") = 0.0, py::arg("k2") = 0.0
    );

  py::class_<Session>(m, "Session")
    .def(py::init<Config, bool>(), py::arg("config"), py::arg("sync") = false)
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("get_feature_points", &Session::getFeaturePoints)
    .def("get_position_cv", &Session::getTwc)
    .def("get_position_gl", &Session::getTwcGL)
    .def("get_position_three", &Session::getTwcThree)
    .def("get_tracking_visualize", &Session::getTrackingVisualize)
    .def("release", &Session::release)
    .def("cancel", &Session::cancel);
}

#endif // __PYBIND_H__