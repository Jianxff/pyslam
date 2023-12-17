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
        class landmark;
    } // namespace data

    namespace publish
    {
        class map_publisher;
    } // namespace publish

} // namespace PLSLAM


class pcl_serializer
{
public:
    pcl_serializer(const std::shared_ptr<PLSLAM::publish::map_publisher> &map_publisher);
    std::string serialize_map_diff();

private:
    const std::shared_ptr<PLSLAM::publish::map_publisher> map_publisher_;
    std::unique_ptr<std::unordered_map<unsigned int, double>> point_hash_map_;

    double current_pose_hash_ = 0;
    int frame_hash_ = 0;

    inline double get_vec_hash(const PLSLAM::Vec3_t &point){
        return point[0] + point[1] + point[2];
    }

    inline double get_mat_hash(const PLSLAM::Mat44_t &pose){
        return pose(0, 3) + pose(1, 3) + pose(2, 3);
    }

    std::string serialize_as_protobuf(const std::vector<PLSLAM::data::landmark*> &landmarks);

    std::string base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len);
};


#endif // WEB_PUBLISHER_DATA_SERIALIZER_H
