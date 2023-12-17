/**
 * This file is part of Structure PLP-SLAM, originally from OpenVSLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Modified by Fangwen Shu <Fangwen.Shu@dfki.de>
 *
 * If you use this code, please cite the respective publications as
 * listed on the github repository.
 *
 * Structure PLP-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Structure PLP-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Structure PLP-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/data/landmark.h"
#include "core/data/landmark_line.h"
#include "core/data/keyframe.h"
#include "core/data/map_database.h"
#include "core/publish/map_publisher.h"

#include <spdlog/spdlog.h>

namespace PLSLAM
{
    namespace publish
    {

        map_publisher::map_publisher(const std::shared_ptr<config> &cfg, data::map_database *map_db)
            : cfg_(cfg), map_db_(map_db)
        {
            spdlog::debug("CONSTRUCT: publish::map_publisher");

        }

        map_publisher::~map_publisher()
        {
            spdlog::debug("DESTRUCT: publish::map_publisher");
        }

        void map_publisher::set_current_cam_pose(const Mat44_t &cam_pose_cw)
        {
            std::lock_guard<std::mutex> lock(mtx_cam_pose_);
            cam_pose_cw_ = cam_pose_cw;
        }

        Mat44_t map_publisher::get_current_cam_pose()
        {
            std::lock_guard<std::mutex> lock(mtx_cam_pose_);
            return cam_pose_cw_;
        }

        unsigned int map_publisher::get_keyframes(std::vector<data::keyframe *> &all_keyfrms)
        {
            all_keyfrms = map_db_->get_all_keyframes();
            return map_db_->get_num_keyframes();
        }

        unsigned int map_publisher::get_landmarks(std::vector<data::landmark *> &all_landmarks,
                                                  std::set<data::landmark *> &local_landmarks)
        {
            all_landmarks = map_db_->get_all_landmarks();
            const auto _local_landmarks = map_db_->get_local_landmarks();
            local_landmarks = std::set<data::landmark *>(_local_landmarks.begin(), _local_landmarks.end());
            return map_db_->get_num_landmarks();
        }

        std::vector<data::landmark*> map_publisher::get_all_landmarks() {
            return map_db_->get_all_landmarks();
        }

        unsigned int map_publisher::get_landmark_lines(std::vector<data::Line *> &all_landmark_lines)
        {
            all_landmark_lines = map_db_->get_all_landmarks_line();
            return map_db_->get_num_landmarks_line();
        }

        bool map_publisher::using_line_tracking() const
        {
            return map_db_->_b_use_line_tracking;
        }
    } // namespace publish
} // namespace PLSLAM
