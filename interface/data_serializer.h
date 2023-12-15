#ifndef WEB_PUBLISHER_DATA_SERIALIZER_H
#define WEB_PUBLISHER_DATA_SERIALIZER_H

#include "core/type.h"

#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace PLSLAM
{

    class config;

    namespace data
    {
        class keyframe;
        class landmark;
    } // namespace data

    namespace publish
    {
        class map_publisher;
    } // namespace publish

} // namespace PLSLAM


class data_serializer
{
public:
    data_serializer(const std::shared_ptr<PLSLAM::publish::map_publisher> &map_publisher);

    std::string serialize_messages(const std::vector<std::string> &tags, const std::vector<std::string> &messages);

    std::string serialize_map_diff();

    static std::string serialized_reset_signal_;

private:
    const std::shared_ptr<PLSLAM::publish::map_publisher> map_publisher_;
    std::unique_ptr<std::unordered_map<unsigned int, double>> keyframe_hash_map_;
    std::unique_ptr<std::unordered_map<unsigned int, double>> point_hash_map_;

    double current_pose_hash_ = 0;
    int frame_hash_ = 0;

    inline double get_vec_hash(const PLSLAM::Vec3_t &point)
    {
        return point[0] + point[1] + point[2];
    }

    inline double get_mat_hash(const PLSLAM::Mat44_t &pose)
    {
        return pose(0, 3) + pose(1, 3) + pose(2, 3);
    }

    Eigen::Matrix4d convert_tcw_three(const PLSLAM::Mat44_t &pose);

    std::string serialize_as_protobuf(const std::vector<PLSLAM::data::keyframe *> &keyfrms,
                                        const std::vector<PLSLAM::data::landmark *> &all_landmarks,
                                        const std::set<PLSLAM::data::landmark *> &local_landmarks,
                                        const PLSLAM::Mat44_t &current_camera_pose);

    std::string base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len);
};


#endif // WEB_PUBLISHER_DATA_SERIALIZER_H
