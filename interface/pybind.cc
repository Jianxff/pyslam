#include "pybind.h"
#include <filesystem>
#include <set>

const Eigen::Matrix4d MAT_X44D_CV2GL_ = 
  (Eigen::Matrix4d() << 
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1).finished();

/************************************
 * Image stream for realtime process
 ************************************/
ImageStream::ImageStream(bool force_realtime)
 : force_realtime_(force_realtime)
{
  // constructor
}

// add new image to image queue
void ImageStream::addNewImage(cv::Mat& im, double time_ms) {
  if(released_) return;

  std::lock_guard<std::mutex> lock(img_mutex_);
  if(img_stream_.size() > 300) {
    spdlog::warn("ImageStream: drop previous frame after 300 frames freezed");
    img_stream_.pop();
    img_times_.pop();
  }
  img_stream_.push(im);
  img_times_.push(time_ms);

  new_img_abailable_ = true;
}

// get new image from image queue
bool ImageStream::getNewImage(cv::Mat& im, double& time) {
  if(released_) return false;

  std::lock_guard<std::mutex> lock(img_mutex_);
  if(!new_img_abailable_) return false;

  do {
      im = img_stream_.front();
      img_stream_.pop();

      time = img_times_.front();
      img_times_.pop();

      if( !force_realtime_ ) break; // check force_realtime to skip frames

  } while( !img_stream_.empty() );

  new_img_abailable_ = !img_stream_.empty();

  return true;
}

// release 
void ImageStream::release() {
  released_ = true;
}


/************************************
 * Pybind interface for config
 ************************************/
Config::Config(
    const std::string& config_file_path,
    const std::string & vocab_file_path,
    const bool mapping,
    const bool line_track,
    const bool loop_detect,
    const bool viewer,
    const bool loadmap,
    const bool serialize,
    const std::string& map_db,
    const std::string& scene_path,
    const std::string& raw_img_dir
) : vocab_file_path_(vocab_file_path),
    line_track_(line_track),
    mapping_(mapping),
    loop_detect_(loop_detect),
    viewer_(viewer),
    map_db_path_(map_db),
    scene_path_(scene_path),
    raw_image_path_(raw_img_dir),
    serialize_(serialize),
    preload(loadmap)
{
  yaml_node_ = YAML::LoadFile(config_file_path_);
  if(raw_image_path_.back() != '/') raw_image_path_.append("/");
  if(viewer_) {
    #ifndef WITH_PANGOLIN_VIEWER
      spdlog::critical("No pangolin support. Build with -DBUILD_PANGOLIN_VIEWER=ON");
      exit(-1);
    #endif
  }
  
}

Config& Config::fitmodel(const int imwidth, const int imheight) {
  // set model
  yaml_node_["Camera.setup"] = "monocular";
  yaml_node_["Camera.model"] = "perspective";
  yaml_node_["Camera.color_order"] = "RGB";
  // set image params
  yaml_node_["Camera.cols"] = imwidth;
  yaml_node_["Camera.rows"] = imheight;
  // set intrinsics
  yaml_node_["Camera.fx"] = (double)(imheight > imwidth ? imwidth : imheight);
  yaml_node_["Camera.fy"] = (double)(imheight > imwidth ? imwidth : imheight);
  yaml_node_["Camera.cx"] = (double)imwidth / 2.0;
  yaml_node_["Camera.cy"] = (double)imheight / 2.0;
  yaml_node_["Camera.k1"] = 0;
  yaml_node_["Camera.k2"] = 0;
  yaml_node_["Camera.p1"] = 0;
  yaml_node_["Camera.p2"] = 0;
  yaml_node_["Camera.k3"] = 0;
  return *this;
}

std::shared_ptr<PLSLAM::config> Config::instance() {
  if(pconfig_ != nullptr) return pconfig_;
  else {
    pconfig_ = std::make_shared<PLSLAM::config>(
      yaml_node_, config_file_path_
    );
    return pconfig_;
  }
}


/************************************
 * Pybind interface for sfm session
 ************************************/
// constructor
Session::Session(const Config& cfg, bool sync) 
  : cfg_(cfg)
{
  spdlog::set_level(spdlog::level::debug);
  
  // reset settings
  psystem_.reset(new PLSLAM::system(cfg_.instance(), cfg_.vocab_file_path_, cfg_.line_track_));
  pstream_.reset(new ImageStream(!sync));
  pmvs_.reset(new MVS::Scene(psystem_));

  // preload
  if(cfg_.preload)
    psystem_->load_map_database(cfg.map_db_path_);
  if(cfg_.loop_detect_)
    psystem_->enable_loop_detector();
  // start session
  psystem_->startup(!cfg.preload);

  // mapping
  if(cfg.mapping_) psystem_->enable_mapping_module();
  else psystem_->disable_mapping_module();

  // serialize
  pserializer_.reset(new pcl_serializer(psystem_->map_publisher_));

  // run thread
  system_thread_ = std::thread(&Session::run, this);

#ifdef WITH_PANGOLIN_VIEWER
  if(cfg_.viewer_) {
    pviewer_.reset(new pangolin_viewer::viewer(cfg_.instance(), psystem_.get(), psystem_->get_frame_publisher(), psystem_->get_map_publisher()));
    viewer_thread_ = std::thread(&pangolin_viewer::viewer::run, pviewer_.get());
  }
#endif
}

// add track image
void Session::addTrack(py::array_t<uint8_t>& input, double time_ms){
  if(released_) return;

  cv::Mat image = getImageRGB(input).clone();
  // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

  if(time_ms < 0) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    time_ms = (double)ms.count();
  }

  pstream_->addNewImage(image, time_ms);
}

// get feature points
py::array_t<float> Session::getFeaturePoints() {
  if(released_) 
    return py::array_t<float>();

  std::vector<float> data;
  auto keypoints = psystem_->tracker_->curr_frm_.keypts_;
  for(auto& kp : keypoints) {
    data.push_back(kp.pt.x);
    data.push_back(kp.pt.y);
  }

  return py::array_t<float>({(int)data.size()/2, 2}, data.data());
}

// get tracking status
py::array_t<size_t> Session::getTrackingState() {
  size_t data[3] = {0,0,0};

  if(!released_) {
    data[0] = psystem_->tracker_->last_tracking_state_;
    data[1] = psystem_->map_db_->get_num_keyframes();
    data[2] = psystem_->map_db_->get_num_landmarks();
  }
  
  return py::array_t<size_t>({3}, data);
}

// get camera twc
Eigen::Matrix4d Session::getTwc() {
  if(released_) return Eigen::Matrix4d::Identity();
  return Twc_;
}

// get camera twc under openGL coordinate
Eigen::Matrix4d Session::getTwcGL() {
  if(released_) return Eigen::Matrix4d::Identity();

  Eigen::Matrix4d Twc = MAT_X44D_CV2GL_ * Twc_;
  return Twc;
}

Eigen::Matrix4d Session::getTwcThree(){
  if(released_) return Eigen::Matrix4d::Identity();

  Eigen::Matrix4d Twc = MAT_X44D_CV2GL_ * Twc_ * MAT_X44D_CV2GL_;
  return Twc;
}

std::string Session::getMapProtoBuf() {
  if(released_) return "";
  return pserializer_->serialize_map_diff();
}

// dump images
void Session::dumpImages() {
  auto keyframes = psystem_->map_db_->get_all_keyframes();
  for(auto& kf : keyframes) {
    cv::Mat img = kf->get_img_rgb();
    if(img.empty()) continue;
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    std::string filename = cfg_.raw_image_path_ + std::to_string(kf->id_) + ".png";
    cv::imwrite(filename, img);
  }
}

// stop session
py::array_t<size_t> Session::release() {
  size_t data[2] = {0,0};
  if(released_) return py::array_t<size_t>({2}, data);

  pstream_->release();
  
  exit_required_ = true;

#ifdef WITH_PANGOLIN_VIEWER
  if(cfg_.viewer_) {
    pviewer_->request_terminate();
    viewer_thread_.join();
  }
#endif

  psystem_->shutdown();
  system_thread_.join();

  released_ = true;
  
  if(cfg_.serialize_) {
    // save map
    psystem_->save_map_database(cfg_.map_db_path_);
    // save scene
    // dumpImages();
    pmvs_->serialize(cfg_.scene_path_, cfg_.raw_image_path_);
  }

  // save last data
  data[0] = psystem_->map_db_->get_num_keyframes();
  data[1] = psystem_->map_db_->get_num_landmarks();

  // reset
  psystem_.reset();
  pserializer_.reset();
  pstream_.reset();
  pmvs_.reset();

#ifdef WITH_PANGOLIN_VIEWER
  pviewer_.reset();
#endif

  cv::destroyAllWindows();

  return py::array_t<size_t>({2}, data);
}

// cancel session
void Session::cancel() {
  cfg_.serialize_ = false;
  release();
  spdlog::warn("Session cancelled");
}

// run slam thread
void Session::run() {
  cv::Mat img;
  double time;
  while( !exit_required_ ) {
    if(pstream_->getNewImage(img, time)) {
      const Eigen::Matrix4d tcw = psystem_->feed_monocular_frame(img, time);
      Twc_ = tcw.inverse();
    } 
    else  
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 60));
  }
  spdlog::info("Stop Processing New Frame");
}

// get image from webrtc frame
cv::Mat Session::getImageRGB(py::array_t<uint8_t>& input) {
  if(input.ndim() != 3) 
    throw std::runtime_error("get Image : number of dimensions must be 3");
  py::buffer_info buf = input.request();
  cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  return image;
}
