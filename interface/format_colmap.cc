#include "format_colmap.h"


#include "core/camera/perspective.h"

#include "core/data/map_database.h"
#include "core/data/keyframe.h"
#include "core/data/landmark.h"

#include <opencv2/core/eigen.hpp>

#include <spdlog/spdlog.h>

namespace Colmap{

const Eigen::Matrix3d MAT_X33D_CV2GL_ = 
  (Eigen::Matrix3d() << 
    1, 0, 0,
    0, -1, 0,
    0, 0, -1).finished();

const Eigen::Matrix3d MAT_X33D_CV2GL_INVERSE_
   = MAT_X33D_CV2GL_;

const unsigned int KEYFRAME_STEP_BY = 800;

Sparse::Sparse(const std::shared_ptr<PLSLAM::system>& psystem) 
  : psystem_(psystem)
{
  // initialize  
}

void Sparse::serialize(const std::string& work_dir) {
    work_dir_ = work_dir;
    if(work_dir_.back() != '/')  work_dir_ += "/";
    image_dir_ = work_dir_ + "images/";
    sparse_dir_ = work_dir_ + "sparse/";
    // make dir
    std::filesystem::create_directories(image_dir_);
    std::filesystem::create_directories(sparse_dir_);

    // filter
    filterKeyframes();
    filterLandmarks();

    // serialize
    serializeCamera();
    serializeImages();
    serializeMapPoints();

    spdlog::info("converted to colmap format: {}", work_dir_);
    spdlog::info("with {} images and {} points3D", keyframes_.size(), landmarks_.size());
}

void Sparse::filterKeyframes() {
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
    int step = kf_size / KEYFRAME_STEP_BY;
    step = (step == 0 ? 1 : step);
    for(int i = 0; i < kf_size; i += step) {
        // find max definition from each step
        keyframes_.insert_or_assign(keyframes_filter[i]->id_, keyframes_filter[i]);
    }
}

void Sparse::filterLandmarks() {
    std::vector<PLSLAM::data::landmark*> landmarks_origin;
    landmarks_origin = psystem_->map_db_->get_all_landmarks();
    // filter
    for(auto& lm : landmarks_origin) {
        // check status
        if(!lm || lm->will_be_erased()) continue;
        landmarks_.insert_or_assign(lm->id_, lm);
    }
}

void Sparse::serializeCamera() {
    const std::string filepath = sparse_dir_ + "cameras.txt";
    // write to file
    std::ofstream ofs(filepath);
    if(!ofs.is_open()) {
        spdlog::error("Failed to open file: {}", filepath);
        return;
    }
    if(psystem_->camera_->model_type_ != PLSLAM::camera::model_type_t::Perspective) {
        spdlog::error("Only support perspective camera");
        return;
    }
    // write header
    ofs << "# Camera list with one line of data per camera:\n"
        << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        << "# Number of cameras: 1\n";
    // write data
    auto camera = dynamic_cast<PLSLAM::camera::perspective*>(psystem_->camera_);
    ofs << "1 PINHOLE " << camera->cols_ << " " << camera->rows_ << " ";
    ofs << camera->fx_ << " " << camera->fy_ << " " << camera->cx_ << " " << camera->cy_;
    ofs << "\n";
    ofs.close();
}

void Sparse::serializeImages() {
    const std::string filepath = sparse_dir_ + "images.txt";
    // write to file
    std::ofstream ofs(filepath);
    if(!ofs.is_open()) {
        spdlog::error("Failed to open file: {}", filepath);
        return;
    }
    // write header
    ofs << "# Image list with two lines of data per image:\n"
        << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        << "# Number of images: " << keyframes_.size() << "\n";
    // write data
    for(auto& ele : keyframes_) {
        auto kf = ele.second;
        // write image
        const std::string image_name = std::to_string(kf->id_) + ".png";
        cv::Mat img = kf->get_img_rgb();
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        cv::imwrite(image_dir_ + image_name, img);

        // write transform
        ofs << kf->id_ << " ";
        Eigen::Matrix3d rcw = kf->get_rotation();
        Eigen::Vector3d tcw = kf->get_translation();
        Eigen::Quaterniond qvec(rcw);
        ofs << qvec.w() << " " << qvec.x() << " " << qvec.y() << " " << qvec.z() << " ";
        ofs << tcw(0) << " " << tcw(1) << " " << tcw(2) << " ";
        ofs << "1 " << image_name << "\n";
        
        // write points
        auto all_lm = kf->get_landmarks();
        for(int i = 0; i < kf->keypts_.size(); i++) {
            auto& pt = kf->keypts_[i].pt;
            int id_ = -1;
            if(i < all_lm.size()) {
                auto lm = kf->get_landmark(i);
                if(lm && landmarks_.count(lm->id_) > 0) id_ = lm->id_;
            }
            ofs << pt.x << " " << pt.y << " " << id_ << " ";

        }
        ofs << "\n";
    }
    ofs.close();
}

void Sparse::serializeMapPoints() {
    const std::string filepath = sparse_dir_ + "points3D.txt";
    // write to file
    std::ofstream ofs(filepath);
    if(!ofs.is_open()) {
        spdlog::error("Failed to open file: {}", filepath);
        return;
    }
    // write header
    ofs << "# 3D point list with one line of data per point:\n"
        << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        << "# Number of points: " << landmarks_.size() << "\n";
    // write data
    for(auto& lm : landmarks_) {
        // write point
        ofs << lm.second->id_ << " ";
        Eigen::Vector3d pos = lm.second->get_pos_in_world();
        ofs << pos(0) << " " << pos(1) << " " << pos(2) << " ";
        auto color = lm.second->color_;
        ofs << (int)color(0) << " " << (int)color(1) << " " << (int)color(2) << " ";
        ofs << "0 ";
        // write track
        auto observations = lm.second->get_observations();
        for(auto& obs : observations) {
            if(obs.first && keyframes_.count(obs.first->id_) > 0){
                ofs << obs.first->id_ << " " << obs.second << " ";
            }
        }
        ofs << "\n";
    }
    ofs.close();
}

} // namespace Colmap


