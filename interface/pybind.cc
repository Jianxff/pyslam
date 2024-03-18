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
ImageStream::ImageStream(bool sync)
 : sync_(sync)
{
  // constructor
  spdlog::info("ImageStream: sync mode {}", sync_ ? "on" : "off");
}

// add new image to image queue
void ImageStream::addNewImage(cv::Mat& im, double time_ms) {
    if(released_) return;

    std::lock_guard<std::mutex> lock(img_mutex_);
    if(!sync_ && img_stream_.size() > 300) {
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

        if( sync_ ) break; // check if force realtime to skip frames

    } while( !img_stream_.empty() );

    new_img_abailable_ = !img_stream_.empty();

    return true;
}

// release 
void ImageStream::release() {
    released_ = true;
}

// status
bool ImageStream::sync() {
    return sync_;
}

size_t ImageStream::count() {
    return img_stream_.size();
}


/************************************
 * Pybind interface for config
 ************************************/
Config::Config(
    const std::string& config_file_path,
    const std::string & vocab_file_path,
    const bool mapping,
    const bool linetrack,
    const bool loopclose,
    const bool viewer,
    const bool loadmap,
    const bool savemap,
    const std::string& map_db_path,
    const bool convert_colmap,
    const std::string& colmap_dir,
    const bool convert_mvs,
    const std::string& mvs_dir
) : config_file_path_(config_file_path),
    vocab_file_path_(vocab_file_path),
    linetrack_(linetrack),
    mapping_(mapping),
    loopclose_(loopclose),
    viewer_(viewer),
    map_db_path_(map_db_path),
    savemap_(savemap),
    loadmap_(loadmap),
    convert_colmap_(convert_colmap),
    colmap_dir_(colmap_dir),
    convert_mvs_(convert_mvs),
    mvs_dir_(mvs_dir)
{
    yaml_node_ = YAML::LoadFile(config_file_path_);
    if(viewer_) {
        #ifndef WITH_PANGOLIN_VIEWER
            spdlog::critical("No pangolin support. Build with -DBUILD_PANGOLIN_VIEWER=ON");
            exit(-1);
        #endif
    }
  
}

Config& Config::setmodel(
    const int imwidth, const int imheight, 
    const double fx, const double fy, 
    const double k1, const double k2
) {
    // set model
    yaml_node_["Camera.setup"] = "monocular";
    yaml_node_["Camera.model"] = "perspective";
    yaml_node_["Camera.color_order"] = "RGB";
    // set image params
    yaml_node_["Camera.cols"] = imwidth;
    yaml_node_["Camera.rows"] = imheight;
    // set intrinsics
    yaml_node_["Camera.fx"] = (double)fx;
    yaml_node_["Camera.fy"] = (double)fy;
    yaml_node_["Camera.cx"] = (double)imwidth / 2.0;
    yaml_node_["Camera.cy"] = (double)imheight / 2.0;
    yaml_node_["Camera.k1"] = k1;
    yaml_node_["Camera.k2"] = k2;
    yaml_node_["Camera.p1"] = 0;
    yaml_node_["Camera.p2"] = 0;
    yaml_node_["Camera.k3"] = 0;
    return *this; 
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
    yaml_node_["Camera.fx"] = (double)(imheight > imwidth ? imheight : imwidth);
    yaml_node_["Camera.fy"] = (double)(imheight > imwidth ? imheight : imwidth);
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
    psystem_.reset(new PLSLAM::system(cfg_.instance(), cfg_.vocab_file_path_, cfg_.linetrack_));
    pstream_.reset(new ImageStream(sync));
    pmvs_.reset(new MVS::Scene(psystem_));
    pcolmap_.reset(new Colmap::Sparse(psystem_));

    // preload
    if(cfg_.loadmap_)
        psystem_->load_map_database(cfg_.map_db_path_);
    if(cfg_.loopclose_)
        psystem_->enable_loop_detector();
    // start session
    psystem_->startup(!cfg_.loadmap_);

    // mapping
    if(cfg_.mapping_) psystem_->enable_mapping_module();
    else psystem_->disable_mapping_module();

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

// get tracking visualize
py::array_t<uint8_t> Session::getTrackingVisualize() {
    if(released_) 
        return py::array_t<uint8_t>();

    cv::Mat img = psystem_->frame_publisher_->draw_frame(false);
    if(img.empty()) return py::array_t<uint8_t>();

    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return py::array_t<uint8_t>({img.rows, img.cols, 3}, img.data);
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

// stop session
py::array_t<size_t> Session::release() {
    size_t data[2] = {0,0};
    if(released_) return py::array_t<size_t>({2}, data);

    if(pstream_->sync()) {
        while (pstream_->count()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 30));
        }
    }

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
    // map databse
    if(cfg_.savemap_) psystem_->save_map_database(cfg_.map_db_path_);
    // to colmap
    if(cfg_.convert_colmap_) pcolmap_->serialize(cfg_.colmap_dir_);
    // to mvs
    if(cfg_.convert_mvs_) pmvs_->serialize(cfg_.mvs_dir_);

    // save last data
    data[0] = psystem_->map_db_->get_num_keyframes();
    data[1] = psystem_->map_db_->get_num_landmarks();

    // reset
    psystem_.reset();
    pstream_.reset();
    pmvs_.reset();
    pcolmap_.reset();

#ifdef WITH_PANGOLIN_VIEWER
    pviewer_.reset();
#endif

    cv::destroyAllWindows();

    return py::array_t<size_t>({2}, data);
}

// cancel session
void Session::cancel() {
    cfg_.savemap_ = false;
    cfg_.convert_colmap_ = false;
    cfg_.convert_mvs_ = false;
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
