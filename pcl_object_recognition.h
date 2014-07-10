#include "all.hpp"
#include "openni2pcl.hpp"

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef std::tuple<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >, std::vector<pcl::Correspondences>> ClusterType;
typedef std::tuple<float, float> error;
std::clock_t init;


const Eigen::Vector4f subsampling_leaf_size (0.01f, 0.01f, 0.01f, 0.0f);

//Algorithm params
float model_ss (0.005);
float scene_ss (0.005);
float rf_rad (0.02);
float descr_rad (0.05);
float cg_size (0.007);
float cg_thresh (6.0f);
float sac_seg_iter (1000);
float reg_sampling_rate (10);
float sac_seg_distance (0.05);
float reg_clustering_threshold (0.2);
float max_inliers (40000);
float min_scale (0.01);
float min_contrast (0.1f);
float support_size (0.02);
float filter_intensity (0.04);
float descriptor_distance (0.25);
float segmentation_threshold (0.01);
float normal_estimation_search_radius (0.05);
int segmentation_iterations (1000);
int n_octaves (3);
int n_scales_per_octave (2);
int random_scene_samples (1000);
int random_model_samples (1000);
int distance (700);  //kinect cut-off distance
int harris_type (1);
bool narf (false);
bool random_points (false);
bool sift (false);
bool harris (false);
bool fpfh (false);
bool pfh (false);
bool pfhrgb (false);
bool ppf (false);
bool ppfrgb (false);
bool shot (false);
bool ransac (false);
bool ppfe (false);
bool first (true);
bool show_keypoints (false);
bool show_correspondences (true);
bool use_hough (true);
bool to_filter (false);
bool show_filtered (false);
bool remove_outliers (false);
bool use_icp (false);
bool segment (false);
bool error_log(false);

void
ShowHelp (char *file_name)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Real time object recognition - Usage Guide                  *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage " << file_name << " model_filename_list [Options]" << std::endl << std::endl;
  std::cout << "Options" << std::endl;
  std::cout << "     -h                                 Show this help." << std::endl;
  std::cout << "     -ppfe                              Uses ppfe overriding all the other parameters." << std::endl;
  std::cout << "     -show_keypoints                    Show used keypoints." << std::endl;
  std::cout << "     -show_correspondences              Show used correspondences." << std::endl;
  std::cout << "     -filter                            Filter the cloud by color leaving only the points which are close to the model color." << std::endl;
  std::cout << "     -remove_outliers                   Remove ouliers from the scene." << std::endl;
  std::cout << "     -segment                           Segments the objects in the scene removing the major plane." << std::endl;
  std::cout << "     -log                               Saves the pose estimation error in a file called pose_error. " << std::endl;
  std::cout << "     --filter_intensity val             Max distance between colors normalized between 0 and 1 (default 0.02)" << std::endl;
  std::cout << "     --descriptor_distance val          Descriptor max distance to be a match (default 0.25)" << std::endl;
  std::cout << "     --algorithm (hough|gc)             Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --keypoints (narf|sift|uniform|random|harris)    Keypoints detection algorithm (default uniform)." << std::endl;
  std::cout << "     --descriptors (shot|fpfh|pfh|pfhrgb|ppf)          Descriptor type (default shot)." << std::endl;
  std::cout << "     --model_ss val                     Model uniform sampling radius (default 0.005)" << std::endl;
  std::cout << "     --scene_ss val                     Scene uniform sampling radius (default 0.005)" << std::endl;
  std::cout << "     --rf_rad val:                      Hough reference frame radius (default 0.02)" << std::endl;
  std::cout << "     --descr_rad val                    Descriptor radius (default 0.03)" << std::endl;
  std::cout << "     --cg_size val                      Dimension of Hough's bins (default 0.007)" << std::endl;
  std::cout << "     --cg_thresh val                    Minimum number of positive votes for a match (default 6)" << std::endl;
  std::cout << "     --sift_min_scale val               (default 0.01)" << std::endl;
  std::cout << "     --sift_octaves val                 (default 3)" << std::endl;
  std::cout << "     --sift_scales_per_octave val       (default 2)" << std::endl;
  std::cout << "     --sift_min_contrast val            (default 0.3)" << std::endl;
  std::cout << "     --narf_support_size val            (default 0.02)" << std::endl;
  std::cout << "     --sac_seg_iter val                 Max iteration number of the ransac segmentation (default 1000)" << std::endl;
  std::cout << "     --reg_clustering_threshold val     Registration position clustering threshold (default 0.2)" << std::endl;
  std::cout << "     --reg_sampling_rate val            Ppfe registration sampling rate (default 10)" << std::endl;
  std::cout << "     --sac_seg_distance val             Ransac segmentation distance threshold (default 0.05)" << std::endl;
  std::cout << "     --max_inliers val                  Max number of inliers (default 40000)" << std::endl;
  std::cout << "     --random_scene_samples val         Number of random samples in the scene (default 1000) " << std::endl;
  std::cout << "     --random_model_samples val         Number of random samples in the model (default 1000) " << std::endl;
  std::cout << "     --harris_type val (HARRIS = 1|NOBLE = 2|LOWE = 3|TOMASI = 4|CURVATURE = 5)                 (default HARRIS) " << std::endl;
  std::cout << "     --descriptor_distance              Maximum distance between descriptors (default 0.25) " << std::endl;
  std::cout << "     --segmentation_threshold           Segmentation threshold for the plane recognition (default 0.01) " << std::endl;
  std::cout << "     --segmentation_iterations          Number of iteration of the segmenter (default 1000) " << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    ShowHelp (argv[0]);
    exit (0);
  }

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-show_keypoints"))
    show_keypoints = true;
  if (pcl::console::find_switch (argc, argv, "-show_correspondences"))
    show_correspondences = true;
  if (pcl::console::find_switch (argc, argv, "-filter"))
    to_filter = true;
  if (pcl::console::find_switch (argc, argv, "-ppfe"))
    ppfe = true;
  if (pcl::console::find_switch (argc, argv, "-remove_outliers"))
    remove_outliers = true;
  if (pcl::console::find_switch (argc, argv, "-segment"))
    segment = true;
  if (pcl::console::find_switch (argc, argv, "-log"))
    error_log = true;

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("hough") == 0)
      use_hough = true;
    else if (used_algorithm.compare ("gc") == 0)
      use_hough = false;
    else
    {
      std::cout << "Wrong algorithm name.\n";
      ShowHelp (argv[0]);
      exit (-1);
    }
  }

  std::string used_keypoints;
  if (pcl::console::parse_argument (argc, argv, "--keypoints", used_keypoints) != -1)
  {
    if (used_keypoints.compare ("narf") == 0)
      narf = true;
    else if (used_keypoints.compare ("sift") == 0)
      sift = true;
    else if (used_keypoints.compare ("ransac") == 0)
      ransac = true;
    else if (used_keypoints.compare ("random") == 0)
      random_points = true;
    else if (used_keypoints.compare ("harris") == 0)
      harris = true;
    else if (used_keypoints.compare ("uniform") == 0)
      std::cout << "Using uniform sampling.\n";

  }

  std::string used_descriptors;
  if (pcl::console::parse_argument (argc, argv, "--descriptors", used_descriptors) != -1)
  {
    if (used_descriptors.compare ("shot") == 0)
      shot = true;
    else if (used_descriptors.compare ("fpfh") == 0)
      fpfh = true;
    else if (used_descriptors.compare ("ppf") == 0)
      ppf = true;
    else if (used_descriptors.compare ("ppfrgb") == 0)
      ppfrgb = true;
    else if (used_descriptors.compare ("pfh") == 0)
      pfh = true;
    else if (used_descriptors.compare ("pfhrgb") == 0)
      pfhrgb = true;
    else
    {
      std::cout << "Wrong descriptors type .\n";
      ShowHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh);
  pcl::console::parse_argument (argc, argv, "--sift_min_scale", min_scale);
  pcl::console::parse_argument (argc, argv, "--sift_octaves", n_octaves);
  pcl::console::parse_argument (argc, argv, "--sift_scales_per_octave", n_scales_per_octave);
  pcl::console::parse_argument (argc, argv, "--sift_min_contrast", min_contrast);
  pcl::console::parse_argument (argc, argv, "--narf_support_size", support_size);
  pcl::console::parse_argument (argc, argv, "--descriptor_distance", descriptor_distance);
  pcl::console::parse_argument (argc, argv, "--max_inliers", max_inliers);
  pcl::console::parse_argument (argc, argv, "--sac_seg_iter", sac_seg_iter);
  pcl::console::parse_argument (argc, argv, "--reg_clustering_threshold", reg_clustering_threshold);
  pcl::console::parse_argument (argc, argv, "--reg_sampling_rate", reg_sampling_rate);
  pcl::console::parse_argument (argc, argv, "--sac_seg_distance", sac_seg_distance);
  pcl::console::parse_argument (argc, argv, "--random_model_samples", random_model_samples);
  pcl::console::parse_argument (argc, argv, "--random_scene_samples", random_scene_samples);
  pcl::console::parse_argument (argc, argv, "--filter_intensity", filter_intensity);
  pcl::console::parse_argument (argc, argv, "--harris_type", harris_type);
  pcl::console::parse_argument (argc, argv, "--descriptor_distance", descriptor_distance);
  pcl::console::parse_argument (argc, argv, "--segmentation_threshold", segmentation_threshold);
  pcl::console::parse_argument (argc, argv, "--segmentation_iterations", segmentation_iterations);

}

inline void
ShowKeyHelp ()
{
  std::cout << "Press q to increase the Hough thresh by 1" << std::endl;
  std::cout << "Press w to decrease the Hough thresh by 1" << std::endl;
  std::cout << "Press a to increase Hough bin size by 0.001" << std::endl;
  std::cout << "Press s to decrease Hough bin size by 0.001" << std::endl;
  std::cout << "Press z to increase the scene sampling size" << std::endl;
  std::cout << "Press x to decrease the scene sampling size" << std::endl;
  std::cout << "Press p to print the actual parameters" << std::endl;
  std::cout << "Press k to toggle filtered mode" << std::endl;
  std::cout << "Press i to toggle icp alignment" << std::endl;
  std::cout << "Press n to incraese aquired distance" << std::endl;
  std::cout << "Press m to incraese aquired distance" << std::endl;
  std::cout << "Press j to switch between filtered and complete view" << std::endl;
  std::cout << "Press d to lower segmentation threshold " << std::endl;
  std::cout << "Press f to increase segmentation threshold" << std::endl;
  std::cout << "Press l to lower filtering" << std::endl;
  std::cout << "Press k to increase filtering" << std::endl;
}

double frobeniusNorm(const Eigen::Matrix3f matrix)
{
    double result = 0.0;
    for(unsigned int i = 0; i < 3; ++i)
    {
        for(unsigned int j = 0; j < 3; ++j)
        {
            double value = matrix(i, j);
            result += value * value;
        }
    }
    return sqrt(result);
}

void
KeyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
  std::string pressed;
  pressed = event.getKeySym ();
  if (event.keyDown ())
  {
    if (pressed == "q")
    {
      cg_thresh++;
      std::cout << "\tcg_thresh increased to " << cg_thresh << std::endl;
    }
    else if (pressed == "w")
    {
      cg_thresh--;
      std::cout << "\tcg_thresh decreased to " << cg_thresh << std::endl;
    }
    else if (pressed == "a")
    {
      cg_size += 0.001;
      std::cout << "\tcg_size increased to " << cg_size << std::endl;
    }
    else if (pressed == "s")
    {
      cg_size -= 0.001;
      std::cout << "\tcg_size decreased to " << cg_size << std::endl;
    }
    else if (pressed == "z")
    {
      scene_ss += 0.001;
      std::cout << "\tscene sampling size increased to " << scene_ss << std::endl;
    }
    else if (pressed == "x")
    {
      scene_ss -= 0.001;
      std::cout << "\tscene sampling size decreased to " << scene_ss << std::endl;
    }
    else if (pressed == "e")
    {
      sac_seg_distance += 0.001;
      std::cout << "\t sac segmentation distance increased to " << sac_seg_distance << std::endl;
    }
    else if (pressed == "e")
    {
      sac_seg_distance -= 0.001;
      std::cout << "\t sac segmentation distance decreased to " << sac_seg_distance << std::endl;
    }
    else if (pressed == "j")
    {
      show_filtered = !show_filtered;
    }
    else if (pressed == "n")
    {
      distance += 100;
    }
    else if (pressed == "d")
    {
      segmentation_threshold += 0.01;
    }
    else if (pressed == "f")
    {
      if (segmentation_threshold > 0)
        segmentation_threshold -= 0.01;
    }
    else if (pressed == "l")
    {
      filter_intensity += 0.01;
    }
    else if (pressed == "k")
    {
      if (filter_intensity > 0)
        filter_intensity -= 0.01;
    }
    else if (pressed == "m")
    {
      if (distance > 200)
        distance -= 100;
    }
    else if (pressed == "i")
    {
      use_icp = !use_icp;
    }
    else if (pressed == "h")
    {
      ShowKeyHelp ();
    }
  }
}

inline void
SetViewPoint (pcl::PointCloud<PointType>::Ptr cloud)
{

  cloud->sensor_origin_.setZero ();
  cloud->sensor_orientation_.w () = 0.0;
  cloud->sensor_orientation_.x () = 1.0;
  cloud->sensor_orientation_.y () = 0.0;
  cloud->sensor_orientation_.z () = 0.0;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
ReadModels (char** argv)
{
  pcl::PCDReader reader;

  std::vector < pcl::PointCloud < pcl::PointXYZRGB > ::Ptr > cloud_models;

  ifstream pcd_file_list (argv[1]);
  while (!pcd_file_list.eof ())
  {
    char str[512];
    pcd_file_list.getline (str, 512);
    if (std::strlen (str) > 2)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      reader.read (str, *cloud);
      ///SetViewPoint(cloud);
      cloud_models.push_back (cloud);
      PCL_INFO ("Model read: %s\n", str);
    }
  }
  std::cout << "all loaded" << std::endl;
  return (std::move (cloud_models));
}

void
PrintTransformation (ClusterType cluster)
{
  for (size_t i = 0; i < std::get < 0 > (cluster).size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << std::get < 1 > (cluster)[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = std::get < 0 > (cluster)[i].block<3, 3> (0, 0);
    Eigen::Vector3f translation = std::get < 0 > (cluster)[i].block<3, 1> (0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0, 0), rotation (0, 1), rotation (0, 2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1, 0), rotation (1, 1), rotation (1, 2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2, 0), rotation (2, 1), rotation (2, 2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }
}

pcl::PointCloud<PointType>::Ptr
FindAndSubtractPlane (const pcl::PointCloud<PointType>::Ptr input, float distance_threshold, float max_iterations)
{
  // Find the dominant plane
  pcl::SACSegmentation<PointType> seg;
  seg.setOptimizeCoefficients (false);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (distance_threshold);
  seg.setMaxIterations (max_iterations);
  seg.setInputCloud (input);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  seg.segment (*inliers, *coefficients);

  // Extract the inliers
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud (input);
  extract.setIndices (inliers);
  extract.setNegative (true);
  pcl::PointCloud<PointType>::Ptr output (new pcl::PointCloud<PointType> ());
  extract.filter (*output);

  return (output);
}

class ColorSampling
{
  public:

    ColorSampling ()
    {
      Clear ();
      rgb2yuv << 0.229, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001;
    }

    void
    Clear ()
    {
      avg_u_ = 0, avg_v_ = 0;
    }

    void
    AddCloud (const pcl::PointCloud<pcl::PointXYZRGB> &cloud)
    {
      Clear ();
      float u, v;
      for (auto point : cloud.points)
      {
        RGBtoYUV (point, u, v);
        avg_u_ += u;
        avg_v_ += v;
      }
      avg_u_ = avg_u_ / cloud.points.size ();
      avg_v_ = avg_v_ / cloud.points.size ();
    }

    void
    FilterPointCloud (const pcl::PointCloud<pcl::PointXYZRGB> &in_cloud, pcl::PointCloud<pcl::PointXYZRGB> &out_cloud)
    {
      pcl::PointCloud < pcl::PointXYZRGB > cloud;
      int points = in_cloud.points.size ();

      for (auto point : in_cloud.points)
      {
        if (!ToFilter (point))
          cloud.points.push_back (point);
      }
      out_cloud = cloud;
      std::cout << "Point number: \n\t Original point cloud: " << points << " \n\t Filtered point cloud: " << cloud.points.size () << std::endl;
    }

    void
    PrintColorInfo ()
    {
      std::cout << "avg U: " << avg_u_ << " V: " << avg_v_ << std::endl;
    }

    Eigen::Matrix3f rgb2yuv;

    float avg_u_, avg_v_;

    bool
    ToFilter (const pcl::PointXYZRGB &point)
    {
      float u, v;
      RGBtoYUV (point, u, v);
      float distance = sqrt (pow (u - avg_u_, 2) + pow (v - avg_v_, 2));
      if (distance < filter_intensity)
        return (false);
      else
        return (true);
    }

    void
    RGBtoYUV (const pcl::PointXYZRGB &point, float &u, float &v)
    {
      Eigen::Vector3f rgb ((float) point.r / 255, (float) point.g / 255, (float) point.b / 255);
      Eigen::Vector3f yuv = rgb2yuv * rgb;
      u = yuv.y ();
      v = yuv.z ();
    }
};

pcl::PointCloud<pcl::PointNormal>::Ptr
SubsampleAndCalculateNormals (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_subsampled (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::VoxelGrid < pcl::PointXYZ > subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (subsampling_leaf_size);
  subsampling_filter.filter (*cloud_subsampled);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_subsampled_normals (new pcl::PointCloud<pcl::Normal> ());
  pcl::NormalEstimation < pcl::PointXYZ, pcl::Normal > normal_estimation_filter;
  normal_estimation_filter.setInputCloud (cloud_subsampled);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr search_tree (new pcl::search::KdTree<pcl::PointXYZ>);
  normal_estimation_filter.setSearchMethod (search_tree);
  normal_estimation_filter.setRadiusSearch (normal_estimation_search_radius);
  normal_estimation_filter.compute (*cloud_subsampled_normals);

  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_subsampled_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
  concatenateFields (*cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);

  PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
  return (cloud_subsampled_with_normals);
}

error
GetRototraslationError (const Eigen::Matrix4f transformation)
{
  Eigen::Matrix3f rotation;
  Eigen::Vector3f traslation;
  float rotation_error;
  float traslation_error;
  error e;
  Eigen::Matrix3f id;
  id.Identity();

  rotation << transformation(0,0), transformation(0,1), transformation(0,2),
              transformation(1,0), transformation(1,1), transformation(1,2),
              transformation(2,0), transformation(2,1), transformation(2,2);  
  traslation << transformation(0,3), transformation(1,3), transformation(2,3);

  traslation_error = traslation.norm();
  std::get < 1 > (e) = traslation_error;
  std::cout << rotation << std::endl;

  Eigen::Matrix3f tmp = id * rotation.transpose();
  float theta = acos((tmp.trace() - 1) / 2);
  rotation_error = frobeniusNorm((theta / (2 * sin(theta) )) * ( tmp - tmp.transpose()) );
  std::get < 0 > (e) = rotation_error;
  std::cout << "theta: " << theta << " (theta / (2 * sin(theta) ) " << theta / (2 * sin(theta) ) << std::endl;
  std::cout << tmp << std::endl;

  return (e);
}

class ErrorWriter
{
public:
  std::ofstream es_;

  ErrorWriter() 
  {
    es_.open("pose_error.txt");
  }

  void 
  WriteError(error e, float fitness)
  { if( std::isnan(std::get < 0 > (e)))
      WriteError(fitness);
    else
      es_ << std::get < 0 > (e) << " " << std::get < 1 > (e) << " " << double(std::clock() - init) / CLOCKS_PER_SEC << " " << fitness << std::endl;
  }

  void 
  WriteError(float fitness)
  {
    es_ << "onf onf "  << double(std::clock() - init) / CLOCKS_PER_SEC << " " << fitness << std::endl;
  }

};

template<class T, class Estimator>
class KeyDes
{
  public:
    typedef pcl::PointCloud<T> PD;
    typedef pcl::PointCloud<PointType> P;
    typedef pcl::PointCloud<NormalType> PN;
    typename PD::Ptr model_descriptors_;
    typename PD::Ptr scene_descriptors_;
    typename P::Ptr model_;
    typename P::Ptr model_keypoints_;
    typename P::Ptr scene_;
    typename P::Ptr scene_keypoints_;
    typename PN::Ptr model_normals_;
    typename PN::Ptr scene_normals_;
    bool created;

    KeyDes (P::Ptr model, P::Ptr model_keypoints, P::Ptr scene, P::Ptr scene_keypoints, PN::Ptr model_normals, PN::Ptr scene_normals) :
        model_descriptors_ (new PD ()), scene_descriptors_ (new PD ()), model_ (model), model_keypoints_ (model_keypoints), scene_ (scene), scene_keypoints_ (scene_keypoints), model_normals_ (model_normals), scene_normals_ (scene_normals), created (false)
    {
    }

    pcl::CorrespondencesPtr
    Run ()
    {

      pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

      //create scene descriptors
      std::cout << "calculating scene descriptors " << std::endl;
      Estimator est;
      est.setInputCloud (scene_keypoints_);
      est.setSearchSurface (scene_);
      est.setInputNormals (scene_normals_);

      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      est.setSearchMethod (tree);
      est.setRadiusSearch (descr_rad);
      est.compute (*scene_descriptors_);

      if (!created)
      {
        //create model descriptors
        std::cout << "calculating model descriptors " << std::endl;
        est.setInputCloud (model_keypoints_);
        est.setSearchSurface (model_);
        est.setInputNormals (model_normals_);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGB>);
        est.setSearchMethod (tree2);
        est.compute (*model_descriptors_);
        created = true;
      }

      pcl::KdTreeFLANN<T> match_search;
      //std::cout <<"calculated " << model_descriptors_->size() << " for the model and " << scene_descriptors_->size() << " for the scene" <<std::endl;

      //  Find Model-Scene Correspondences with KdTree
      std::cout << "calculating correspondences " << std::endl;

      match_search.setInputCloud (model_descriptors_);

      //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
      #pragma omp parallel for 
      for (size_t i = 0; i < scene_descriptors_->size (); ++i)
      {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (match_search.point_representation_->isValid (scene_descriptors_->at (i)))
        {
          int found_neighs = match_search.nearestKSearch (scene_descriptors_->at (i), 1, neigh_indices, neigh_sqr_dists);
          if (found_neighs == 1 && neigh_sqr_dists[0] < descriptor_distance)
          {
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            #pragma omp critical
            model_scene_corrs->push_back (corr);
          }
        }
      }
      /*
       pcl::registration::CorrespondenceRejectorSampleConsensus<PointType> rejector;
       rejector.setInputSource(model_);
       rejector.setInputTarget(scene_);
       rejector.setInputCorrespondences(model_scene_corrs_);
       rejector.getCorrespondences(*model_scene_corrs_);
       */
      std::cout << "\tFound " << model_scene_corrs->size () << " correspondences " << std::endl;
      return (model_scene_corrs);

    }
};

class Ppfe
{
  public:

    pcl::SACSegmentation<pcl::PointXYZ> seg_;
    pcl::ExtractIndices<pcl::PointXYZ> extract_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_xyz_;
    pcl::ModelCoefficients::Ptr coefficients_;
    pcl::PointIndices::Ptr inliers_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_model_input_;
    pcl::PointCloud<pcl::PPFSignature>::Ptr cloud_model_ppf_;
    pcl::PPFRegistration<pcl::PointNormal, pcl::PointNormal> ppf_registration_;
    pcl::PPFHashMapSearch::Ptr hashmap_search_;
    unsigned nr_points_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_scene_input_;
    pcl::PointCloud<pcl::PointNormal> cloud_output_subsampled_;
    Eigen::Matrix4f mat_;
    ClusterType cluster_;

    Ppfe (pcl::PointCloud<PointType>::Ptr model)
    {

      hashmap_search_ = boost::make_shared < pcl::PPFHashMapSearch > (12.0f / 180.0f * float (M_PI), 0.05f);
      cloud_model_ppf_ = boost::make_shared<pcl::PointCloud<pcl::PPFSignature>> ();
      inliers_ = boost::make_shared<pcl::PointIndices> ();
      model_xyz_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>> ();
      copyPointCloud (*model, *model_xyz_);
      coefficients_ = boost::make_shared<pcl::ModelCoefficients> ();
      cloud_scene_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>> ();
      seg_.setOptimizeCoefficients (true);
      seg_.setModelType (pcl::SACMODEL_PLANE);
      seg_.setMethodType (pcl::SAC_RANSAC);
      seg_.setMaxIterations (sac_seg_iter);
      extract_.setNegative (true);
      ppf_registration_.setSceneReferencePointSamplingRate (reg_sampling_rate);  //10
      ppf_registration_.setPositionClusteringThreshold (reg_clustering_threshold);  //0.2f
      ppf_registration_.setRotationClusteringThreshold (30.0f / 180.0f * float (M_PI));
      cloud_model_input_ = SubsampleAndCalculateNormals (model_xyz_);
      pcl::PPFEstimation < pcl::PointNormal, pcl::PointNormal, pcl::PPFSignature > ppf_estimator;
      ppf_estimator.setInputCloud (cloud_model_input_);
      ppf_estimator.setInputNormals (cloud_model_input_);
      ppf_estimator.compute (*cloud_model_ppf_);
      hashmap_search_->setInputFeatureCloud (cloud_model_ppf_);
      ppf_registration_.setSearchMethod (hashmap_search_);
      ppf_registration_.setInputSource (cloud_model_input_);
    }

    ClusterType
    GetCluster (pcl::PointCloud<PointType>::Ptr scene)
    {
      seg_.setDistanceThreshold (sac_seg_distance);
      copyPointCloud (*scene, *cloud_scene_);
      nr_points_ = unsigned (cloud_scene_->points.size ());
      while (cloud_scene_->points.size () > 0.3 * nr_points_)
      {
        seg_.setInputCloud (cloud_scene_);
        seg_.segment (*inliers_, *coefficients_);
        PCL_INFO ("Plane inliers: %u\n", inliers_->indices.size ());
        if (inliers_->indices.size () < max_inliers)
          break;
        extract_.setInputCloud (cloud_scene_);
        extract_.setIndices (inliers_);
        extract_.filter (*cloud_scene_);
      }
      cloud_scene_input_ = SubsampleAndCalculateNormals (cloud_scene_);

      ppf_registration_.setInputTarget (cloud_scene_input_);
      ppf_registration_.align (cloud_output_subsampled_);
      // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output_subsampled_xyz (new pcl::PointCloud<pcl::PointXYZ> ());
      //for (size_t i = 0; i < cloud_output_subsampled_.points.size (); ++i)
      //  cloud_output_subsampled_xyz->points.push_back ( pcl::PointXYZ (cloud_output_subsampled_.points[i].x, cloud_output_subsampled_.points[i].y, cloud_output_subsampled_.points[i].z));
      mat_ = ppf_registration_.getFinalTransformation ();
      std::vector < pcl::Correspondences > cor_tmp;

      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4, 0, 4, 4>>>mat_tmp;
      mat_tmp.push_back (mat_);
      ClusterType cluster_ = std::make_tuple (mat_tmp, cor_tmp);
      inliers_->indices.clear ();
      return (cluster_);
    }

    pcl::PointCloud<PointType>::Ptr
    GetModelKeypoints ()
    {
      pcl::PointCloud<PointType>::Ptr tmp (new pcl::PointCloud<PointType> ());
      copyPointCloud (*cloud_model_input_, *tmp);
      return (tmp);
    }

    pcl::PointCloud<PointType>::Ptr
    GetSceneKeypoints ()
    {
      pcl::PointCloud<PointType>::Ptr tmp (new pcl::PointCloud<PointType> ());
      copyPointCloud (*cloud_scene_, *tmp);
      return (tmp);
    }

};

class OpenniStreamer
{
  public:
    openni::Device device_;        // Software object for the physical device i.e.  
    openni::VideoStream ir_;       // IR VideoStream Class Object
    openni::VideoStream color_;    // Color VideoStream Class Object
    openni::Status rc_;

    OpenniStreamer ()
    {
      rc_ = openni::OpenNI::initialize ();  // Initialize OpenNI 
      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "OpenNI initialization failed" << std::endl;
        openni::OpenNI::shutdown ();
      }
      else
        std::cout << "OpenNI initialization successful" << std::endl;

      rc_ = device_.open (openni::ANY_DEVICE);
      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "Device initialization failed" << std::endl;
        device_.close ();
      }
      rc_ = ir_.create (device_, openni::SENSOR_DEPTH);    // Create the VideoStream for IR

      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "Ir sensor creation failed" << std::endl;
        ir_.destroy ();
      }
      else
        std::cout << "Ir sensor creation successful" << std::endl;
      rc_ = ir_.start ();                      // Start the IR VideoStream
      //ir.setMirroringEnabled(TRUE); 
      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "Ir activation failed" << std::endl;
        ir_.destroy ();
      }
      else
        std::cout << "Ir activation successful" << std::endl;

      device_.setImageRegistrationMode (openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

      //ir.setImageRegistrationMode(ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR);
      rc_ = color_.create (device_, openni::SENSOR_COLOR);    // Create the VideoStream for Color

      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "Color sensor creation failed" << std::endl;
        color_.destroy ();
      }
      else
        std::cout << "Color sensor creation successful" << std::endl;
      rc_ = color_.start ();                      // Start the Color VideoStream

      if (rc_ != openni::STATUS_OK)
      {
        std::cout << "Color sensor activation failed" << std::endl;
        color_.destroy ();
      }
      else
        std::cout << "Color sensor activation successful" << std::endl;
    }
};

class NormalEstimator
{
  public:

    pcl::NormalEstimationOMP<PointType, NormalType> norm_est_;

    NormalEstimator ()
    {
      norm_est_.setKSearch (10);
    }

    NormalEstimator (int n_neighbours) :
        NormalEstimator ()
    {
      norm_est_.setKSearch (n_neighbours);
    }

    pcl::PointCloud<NormalType>::Ptr
    Get_normals (pcl::PointCloud<PointType>::Ptr cloud)
    {
      pcl::PointCloud<NormalType>::Ptr normals (new pcl::PointCloud<NormalType> ());
      norm_est_.setInputCloud (cloud);
      norm_est_.compute (*normals);
      return (normals);
    }
};

class DownSampler
{
  public:
    pcl::VoxelGrid<pcl::PointXYZRGB> down_sampler_;

    DownSampler ()
    {
      down_sampler_.setLeafSize (0.001, 0.001, 0.001);
    }

    DownSampler (float x, float y, float z)
    {
      down_sampler_.setLeafSize (x, y, z);
    }

    void
    SetSampleSize (float x, float y, float z)
    {
      down_sampler_.setLeafSize (x, y, z);
    }

    void
    DownSample (pcl::PointCloud<PointType>::Ptr cloud)
    {
      down_sampler_.setInputCloud (cloud);
      down_sampler_.filter (*cloud);
    }
};

class Narf
{
  public:
    pcl::PointCloud<int> cloud_keypoint_indices_;
    Eigen::Affine3f cloud_sensor_pose_;
    bool rotation_invariant_;
    pcl::RangeImageBorderExtractor range_image_border_extractor_;
    pcl::NarfKeypoint narf_keypoint_detector_;

    Narf () :
        rotation_invariant_ (true), cloud_sensor_pose_ (Eigen::Affine3f::Identity ())
    {
      narf_keypoint_detector_.setRangeImageBorderExtractor (&range_image_border_extractor_);
      narf_keypoint_detector_.getParameters ().support_size = support_size;

    }

    void
    GetKeypoints (pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints)
    {

      boost::shared_ptr < pcl::RangeImage > cloud_range_image_ptr_ (new pcl::RangeImage);

      cloud_sensor_pose_ = Eigen::Affine3f (Eigen::Translation3f (cloud->sensor_origin_[0], cloud->sensor_origin_[1], cloud->sensor_origin_[2])) * Eigen::Affine3f (cloud->sensor_orientation_);

      pcl::RangeImage& cloud_range_image_ = *cloud_range_image_ptr_;

      narf_keypoint_detector_.setRangeImage (&cloud_range_image_);

      cloud_range_image_.createFromPointCloud (*cloud, pcl::deg2rad (0.5f), pcl::deg2rad (360.0f), pcl::deg2rad (180.0f), cloud_sensor_pose_, pcl::RangeImage::CAMERA_FRAME, 0.0, 0.0f, 1);

      cloud_range_image_.setUnseenToMaxRange ();

      narf_keypoint_detector_.compute (cloud_keypoint_indices_);

      cloud_keypoints->points.resize (cloud_keypoint_indices_.points.size ());

      #pragma omp parallel for
      for (size_t i = 0; i < cloud_keypoint_indices_.points.size (); ++i)
        cloud_keypoints->points[i].getVector3fMap () = cloud_range_image_.points[cloud_keypoint_indices_.points[i]].getVector3fMap ();
    }
};

class Sift
{
  public:
    pcl::PointCloud<pcl::PointWithScale> cloud_result_;
    pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_;

    Sift () :
        tree_ (new pcl::search::KdTree<pcl::PointXYZRGB> ())
    {
      sift_.setSearchMethod (tree_);
      sift_.setScales (min_scale, n_octaves, n_scales_per_octave);
      sift_.setMinimumContrast (min_contrast);
    }

    void
    GetKeypoints (pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints)
    {

      sift_.setInputCloud (cloud);
      sift_.compute (cloud_result_);
      copyPointCloud (cloud_result_, *cloud_keypoints);
    }
};

class Harris
{
  public:
    pcl::HarrisKeypoint3D<PointType, pcl::PointXYZI>* harris3D_;

    Harris () :
        harris3D_ (new pcl::HarrisKeypoint3D<PointType, pcl::PointXYZI> (pcl::HarrisKeypoint3D<PointType, pcl::PointXYZI>::HARRIS))
    {
      harris3D_->setNonMaxSupression (true);
      harris3D_->setRadius (0.03);
      harris3D_->setRadiusSearch (0.03);
      switch (harris_type)
      {
        default:
          harris3D_->setMethod (pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::HARRIS);
          break;
        case 2:
          harris3D_->setMethod (pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::NOBLE);
          break;
        case 3:
          harris3D_->setMethod (pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::LOWE);
          break;
        case 4:
          harris3D_->setMethod (pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::TOMASI);
          break;
        case 5:
          harris3D_->setMethod (pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::CURVATURE);
          break;
      }
    }

    void
    GetKeypoints (pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints)
    {
      harris3D_->setInputCloud (cloud);
      pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_temp (new pcl::PointCloud<pcl::PointXYZI>);
      harris3D_->compute (*keypoints_temp);
      copyPointCloud (*keypoints_temp, *cloud_keypoints);
    }
};

template<class T>
class Ransac
{
  public:
    std::vector<int> cloud_inliers_;

    void
    GetKeypoints (pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints)
    {

      typename T::Ptr cloud_plane (new T (cloud));

      pcl::RandomSampleConsensus < pcl::PointXYZRGB > model_ransac (cloud_plane);
      model_ransac.computeModel ();
      model_ransac.getInliers (cloud_inliers_);

      pcl::copyPointCloud < pcl::PointXYZRGB > (*cloud, cloud_inliers_, *cloud_keypoints);
    }

};

class Uniform
{
  public:
    pcl::UniformSampling<PointType> uniform_sampling_;
    pcl::PointCloud<int> sampled_indices_;
    float cloud_ss_ = 0;

    void
    SetSamplingSize (float sampling_size)
    {
      cloud_ss_ = sampling_size;
    }

    void
    GetKeypoints (pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints)
    {
      if (cloud_ss_ != 0)
      {
        uniform_sampling_.setInputCloud (cloud);
        uniform_sampling_.setRadiusSearch (cloud_ss_);
        uniform_sampling_.compute (sampled_indices_);
        pcl::copyPointCloud (*cloud, sampled_indices_.points, *cloud_keypoints);
      }
      else
        std::cout << "no sampling size inserted" << std::endl;
    }

};

class Hough
{
  public:
    ClusterType cluster_;
    pcl::PointCloud<RFType>::Ptr model_rf_;
    pcl::PointCloud<RFType>::Ptr scene_rf_;
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est_;
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer_;
    bool created_;

    Hough () :
        model_rf_ (new pcl::PointCloud<RFType> ()), scene_rf_ (new pcl::PointCloud<RFType> ()), created_ (false)
    {
      rf_est_.setFindHoles (true);
      rf_est_.setRadiusSearch (rf_rad);
      clusterer_.setHoughBinSize (cg_size);
      clusterer_.setHoughThreshold (cg_thresh);
      clusterer_.setUseInterpolation (true);
      clusterer_.setUseDistanceWeight (false);

    }

    ClusterType
    GetClusters (pcl::PointCloud<PointType>::Ptr model, pcl::PointCloud<PointType>::Ptr model_keypoints, pcl::PointCloud<NormalType>::Ptr model_normals, pcl::PointCloud<PointType>::Ptr scene, pcl::PointCloud<PointType>::Ptr scene_keypoints,
        pcl::PointCloud<NormalType>::Ptr scene_normals, pcl::CorrespondencesPtr model_scene_corrs)
    {

      clusterer_.setHoughBinSize (cg_size);
      clusterer_.setHoughThreshold (cg_thresh);
      //  Compute (Keypoints) Reference Frames only for Hough
      if (!created_)
      {
        rf_est_.setInputCloud (model_keypoints);
        rf_est_.setInputNormals (model_normals);
        rf_est_.setSearchSurface (model);
        rf_est_.compute (*model_rf_);
      }

      //std::cout << "computed hough BOARD on model" <<std::endl;

      rf_est_.setInputCloud (scene_keypoints);
      rf_est_.setInputNormals (scene_normals);
      rf_est_.setSearchSurface (scene);
      rf_est_.compute (*scene_rf_);

      //std::cout << "computed hough BOARD on scene" <<std::endl;

      //  Clustering
      if (!created_)
      {
        clusterer_.setInputCloud (model_keypoints);
        clusterer_.setInputRf (model_rf_);
        created_ = true;
      }

      clusterer_.setSceneCloud (scene_keypoints);
      clusterer_.setSceneRf (scene_rf_);
      clusterer_.setModelSceneCorrespondences (model_scene_corrs);

      //std::cout << "prepared Hough for clustering" <<std::endl;

      //clusterer_.cluster_ (clustered_corrs);
      clusterer_.recognize (std::get < 0 > (cluster_), std::get < 1 > (cluster_));
      return (cluster_);
    }

};

class GCG
{
  public:
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer_;
    ClusterType cluster_;

    GCG ()
    {
      gc_clusterer_.setGCSize (cg_size);
      gc_clusterer_.setGCThreshold (cg_thresh);
    }

    ClusterType
    GetClusters (pcl::PointCloud<PointType>::Ptr model_keypoints, pcl::PointCloud<PointType>::Ptr scene_keypoints, pcl::CorrespondencesPtr model_scene_corrs)
    {

      gc_clusterer_.setInputCloud (model_keypoints);
      gc_clusterer_.setSceneCloud (scene_keypoints);
      gc_clusterer_.setModelSceneCorrespondences (model_scene_corrs);

      //gc_clusterer_.cluster_ (clustered_corrs);
      gc_clusterer_.recognize (std::get < 0 > (cluster_), std::get < 1 > (cluster_));
      return (cluster_);

    }
};

template<class T, class TT>
class ICPRegistration
{
  public:
    pcl::IterativeClosestPoint<T, TT> icp_;
    Eigen::Matrix4f transformation_;
    Eigen::Matrix3f rotation_;
    Eigen::Vector3f traslation_;
    float fitness_score_;



    ICPRegistration ()
    {
      // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
      icp_.setMaxCorrespondenceDistance (0.05);
      // Set the maximum number of iterations (criterion 1)
      icp_.setMaximumIterations (20);
      // Set the transformation epsilon (criterion 2)
      icp_.setTransformationEpsilon (1e-8);
      // Set the euclidean distance difference epsilon (criterion 3)
      icp_.setEuclideanFitnessEpsilon (1);

      fitness_score_ = 1;

      transformation_.Identity();
    }                 

    void
    Align (typename pcl::PointCloud<T>::Ptr cloud_source, typename pcl::PointCloud<TT>::Ptr cloud_target)
    {

      typename pcl::PointCloud<T>::Ptr cloud_source_registered (new typename pcl::PointCloud<T> ());

      icp_.setInputSource (cloud_source);
      icp_.setInputTarget (cloud_target);

      // Perform the alignment
      icp_.align (*cloud_source);

      // Obtain the transformation that aligned cloud_source to cloud_source_registered
      transformation_ = icp_.getFinalTransformation ();
      fitness_score_ = icp_.getFitnessScore();
    }
};

class Visualizer
{
  public:
    pcl::visualization::PCLVisualizer viewer_;
    pcl::PointCloud<PointType>::Ptr off_scene_model_;
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints_;
    ICPRegistration<PointType, PointType> icp_;
    int iter_;
    bool clean_;
    ErrorWriter e;
    int r, g;
    std::stringstream ss_cloud_;
    std::stringstream ss_line_;
    std::vector<std::string> to_remove_;

    Visualizer () :
        off_scene_model_ (new pcl::PointCloud<PointType> ()), off_scene_model_keypoints_ (new pcl::PointCloud<PointType> ()), iter_ (0), clean_ (true), r(255), g(0)
    {
      viewer_.registerKeyboardCallback (KeyboardEventOccurred);

    }

    void
    Visualize (pcl::PointCloud<PointType>::Ptr model, pcl::PointCloud<PointType>::Ptr model_keypoints, pcl::PointCloud<PointType>::Ptr scene, pcl::PointCloud<PointType>::Ptr scene_keypoints, ClusterType cluster, pcl::PointCloud<PointType>::Ptr filtered_scene)
    {

      if (!clean_)
      {
        for (auto s : to_remove_)
          viewer_.removeShape (s);
        clean_ = true;
        to_remove_.clear ();
      }
      //SetViewPoint(scene);
      pcl::transformPointCloud (*model, *off_scene_model_, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
      if (show_filtered)
      {
        scene = filtered_scene;
      }
      if (iter_ == 0)
      {
        SetViewPoint (off_scene_model_);
        viewer_.addPointCloud (off_scene_model_, "off_scene_model_");
        viewer_.addPointCloud (scene, "scene_cloud");
      }
      else
        viewer_.updatePointCloud (scene, "scene_cloud");

      pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints_, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));

      if (show_keypoints)
      {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
        SetViewPoint (scene_keypoints);
        if (iter_ == 0)
          viewer_.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        else
          viewer_.updatePointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer_.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints_, 0, 0, 255);
        SetViewPoint (off_scene_model_keypoints_);
        if (iter_ == 0)
          viewer_.addPointCloud (off_scene_model_keypoints_, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints_");
        else
          viewer_.updatePointCloud (off_scene_model_keypoints_, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints_");
        viewer_.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints_");
      }

      for (size_t i = 0; i < std::get < 0 > (cluster).size (); ++i)
      {
        clean_ = false;
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*model, *rotated_model, std::get < 0 > (cluster)[i]);
        if (use_icp && filtered_scene->points.size() > 20)
        {
          icp_.Align (rotated_model, filtered_scene);
          //pcl::transformPointCloud (*rotated_model, *rotated_model, transformation);
        }
        SetViewPoint (rotated_model);

        ss_cloud_ << "instance" << i;
        to_remove_.push_back (ss_cloud_.str ());

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, r, g, 0);

        viewer_.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud_.str ());

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        if (show_correspondences)
        {
          for (size_t j = 0; j < std::get < 1 > (cluster)[i].size (); ++j)
          {
            ss_line_ << "correspondence_line" << i << "_" << j << "_" << iter_;
            float model_x = off_scene_model_keypoints_->at (std::get < 1 > (cluster)[i][j].index_query).x;
            float model_y = off_scene_model_keypoints_->at (std::get < 1 > (cluster)[i][j].index_query).y;
            float model_z = off_scene_model_keypoints_->at (std::get < 1 > (cluster)[i][j].index_query).z;
            float scene_x = scene_keypoints->at (std::get < 1 > (cluster)[i][j].index_match).x;
            float scene_y = scene_keypoints->at (std::get < 1 > (cluster)[i][j].index_match).y;
            float scene_z = scene_keypoints->at (std::get < 1 > (cluster)[i][j].index_match).z;

            Eigen::Quaternion<float> transformation (0, 1, 0, 0);
            Eigen::Vector3f tmp (model_x, model_y, model_z);
            tmp = transformation._transformVector (tmp);
            pcl::PointXYZ model_point (tmp.x (), tmp.y (), tmp.z ());

            Eigen::Vector3f tmp2 (scene_x, scene_y, scene_z);
            tmp2 = transformation._transformVector (tmp2);
            pcl::PointXYZ scene_point (tmp2.x (), tmp2.y (), tmp2.z ());

            viewer_.addLine<pcl::PointXYZ, pcl::PointXYZ> (model_point, scene_point, 0, 255, 0, ss_line_.str ());
            to_remove_.push_back (ss_line_.str ());
          }
        }
      }
      if(error_log && use_icp)
      {
        if(icp_.fitness_score_ < 0.00065 && filtered_scene->points.size() > 20)
        {
          g = 255;
          r = 0;
          e.WriteError(GetRototraslationError((icp_.transformation_)), icp_.fitness_score_);
        }
        else 
        {
          r = 255;
          g = 0;
          e.WriteError(icp_.fitness_score_);
        }
      }

      viewer_.spinOnce ();
      iter_++;
    }
};

