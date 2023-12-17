#include "pcl_serializer.h"

#include "core/data/landmark.h"
#include "core/publish/map_publisher.h"

#include <forward_list>

#include <opencv2/imgcodecs.hpp>

// pointcloud.pb.h will be generated into build/intreface/ when make
#include "pointcloud.pb.h"


pcl_serializer::pcl_serializer(const std::shared_ptr<PLSLAM::publish::map_publisher> &map_publisher)
    : map_publisher_(map_publisher),
      point_hash_map_(new std::unordered_map<unsigned int, double>)
{

}

std::string pcl_serializer::serialize_map_diff()
{
    auto all_landmarks = map_publisher_->get_all_landmarks();
    const auto current_camera_pose = map_publisher_->get_current_cam_pose();

    const double pose_hash = get_mat_hash(current_camera_pose);
    if (pose_hash == current_pose_hash_){
        current_pose_hash_ = pose_hash;
        return "";
    }
    current_pose_hash_ = pose_hash;

    return serialize_as_protobuf(all_landmarks);
}


std::string pcl_serializer::serialize_as_protobuf(const std::vector<PLSLAM::data::landmark*> &all_landmarks)
{
    pointcloud map;

    // landmark registration
    std::unordered_map<unsigned int, double> next_point_hash_map;
    for (const auto landmark : all_landmarks){
        if (!landmark || landmark->will_be_erased())
            continue;

        const auto id = landmark->id_;
        const auto pos = landmark->get_pos_in_world();
        const auto zip = get_vec_hash(pos);
        const auto rgb = landmark->color_;

        // point exists on next_point_zip.
        next_point_hash_map[id] = zip;

        // remove point from point_zip.
        if (point_hash_map_->count(id) != 0){
            if (point_hash_map_->at(id) == zip){
                point_hash_map_->erase(id);
                continue;
            }
            point_hash_map_->erase(id);
        }

        // add to protocol buffers
        auto landmark_obj = map.add_points();
        landmark_obj->set_id(id);
        // add points position
        landmark_obj->add_xyz(pos[0]);
        landmark_obj->add_xyz(-pos[1]);
        landmark_obj->add_xyz(-pos[2]);
        // add points color rgb
        for (int i = 0; i < 3; i++)
            landmark_obj->add_rgb(rgb[i]);
    }
    // removed points are remaining in "point_zips".
    for (const auto &itr : *point_hash_map_){
        const auto id = itr.first;
        auto landmark_obj = map.add_points();
        landmark_obj->set_id(id);
    }
    *point_hash_map_ = next_point_hash_map;

    std::string buffer;
    map.SerializeToString(&buffer);

    const auto *cstr = reinterpret_cast<const unsigned char *>(buffer.c_str());
    return base64_encode(cstr, buffer.length());
}

std::string pcl_serializer::base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len)
{
    static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::stringstream ss;
    int i = 0, j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--)
    {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; (i < 4); i++)
                ss << base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        for (j = 0; (j < i + 1); j++)
            ss << base64_chars[char_array_4[j]];
        while ((i++ < 3))
            ss << '=';
    }
    return ss.str();
}

