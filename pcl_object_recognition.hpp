
#include "openni2pcl_reg.hpp"
#include <pcl/registration/icp.h>


typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef std::tuple<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >, std::vector<pcl::Correspondences>> ClusterType;


const Eigen::Vector4f subsampling_leaf_size (0.01f, 0.01f, 0.01f, 0.0f);
const float normal_estimation_search_radius = 0.05f;


//Algorithm params
float model_ss(0.005);  
float scene_ss(0.005);  
float rf_rad_(0.02);
float descr_rad(0.05);
float cg_size(0.007);
float cg_thresh(6.0f);
float descriptor_distance(0.25);
float sac_seg_iter(1000);
float reg_sampling_rate(10);
float sac_seg_distance(0.05);
float reg_clustering_threshold(0.2);
float max_inliers(40000);
float min_scale(0.001);
float min_contrast(0.1f);
float support_size(0.02);
float filter_intensity(0.02);
int n_octaves(6);
int n_scales_per_octave (4);
int random_scene_samples(1000);
int random_model_samples(1000);
int distance(700); //kinect cut-off distance
bool narf(false);
bool random_points(false);
bool sift(false); 
bool fpfh(false); 
bool ransac(false);
bool ppfe(false);
bool first(true);
bool show_keypoints(false);
bool show_correspondences(true);
bool use_hough(true);
bool to_filter(false);
bool show_filtered(false);
bool remove_outliers(false);
bool use_icp(false);


void showHelp (char *filename){
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Real time object recognition - Usage Guide                  *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename_list [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                                Show this help." << std::endl;
  std::cout << "     -ppfe:                             Uses ppfe overriding all the other parameters." << std::endl;
  std::cout << "     -show_keypoints:                   Show used keypoints." << std::endl;
  std::cout << "     -show_correspondences:             Show used correspondences." << std::endl;
  std::cout << "     -filter:                           Filter the cloud by color leaving only the points which are close to the model color." << std::endl;
  std::cout << "     -remove_outliers:                  Remove ouliers from the scene." << std::endl;
  std::cout << "     --filter_intensity:                Max distance between colors normalized between 0 and 1 (default 0.02)" << std::endl;
  std::cout << "     --descriptor_distance:             Descriptor max distance to be a match (default 0.25)" << std::endl;
  std::cout << "     --algorithm (hough|gc):            Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --keypoints (narf|sift|uniform|random):   Keypoints detection algorithm (default uniform)." << std::endl;
  std::cout << "     --descriptors (shot|fpfh):         Descriptor type (default shot)." << std::endl;
  std::cout << "     --model_ss val:                    Model uniform sampling radius (default 0.005)" << std::endl;
  std::cout << "     --scene_ss val:                    Scene uniform sampling radius (default 0.005)" << std::endl;
  std::cout << "     --rf_rad val:                      Hough reference frame radius (default 0.02)" << std::endl;
  std::cout << "     --descr_rad val:                   Descriptor radius (default 0.03)" << std::endl;
  std::cout << "     --cg_size val:                     Dimension of Hough's bins (default 0.007)" << std::endl;
  std::cout << "     --cg_thresh val:                   Minimum number of positive votes for a match (default 6)" << std::endl;
  std::cout << "     --sift_min_scale:                  (default 0.001)" << std::endl;
  std::cout << "     --sift_octaves:                    (default 6)" << std::endl;
  std::cout << "     --sift_scales_per_octave:          (default 4)" << std::endl;
  std::cout << "     --sift_min_contrast:               (default 0.3)" << std::endl << std::endl;
  std::cout << "     --narf_support_size:               (default 0.02)" << std::endl << std::endl;
  std::cout << "     --sac_seg_iter:                    max iteration number of the ransac segmentation (default 1000)" << std::endl;
  std::cout << "     --reg_clustering_threshold         registration position clustering threshold (default 0.2)"  << std::endl;
  std::cout << "     --reg_sampling_rate                ppfe registration sampling rate (default 10)"  << std::endl;
  std::cout << "     --sac_seg_distance                 ransac segmentation distance threshold (default 0.05)"  << std::endl;
  std::cout << "     --max_inliers                      max number of inliers (default 40000)"  << std::endl;
  std::cout << "     --random_scene_samples                   number of random samples in the scene (default 1000) "   << std::endl;
  std::cout << "     --random_model_samples                   number of random samples in the model (default 1000) "   << std::endl;

}


void parseCommandLine (int argc, char *argv[]){
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
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


  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1){
    if (used_algorithm.compare ("hough") == 0)
      use_hough = true;
    else if (used_algorithm.compare ("gc") == 0)
      use_hough = false;
    else{
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  std::string used_keypoints;
  if (pcl::console::parse_argument (argc, argv, "--keypoints", used_keypoints) != -1){
    if (used_keypoints.compare("narf") == 0)
      narf = true;
    else if (used_keypoints.compare("sift") == 0)
      sift = true;
    else if(used_keypoints.compare("ransac") == 0)
      ransac = true;
    else if(used_keypoints.compare("random") == 0)
      random_points = true;
    else if(used_keypoints.compare("uniform") == 0)
      std::cout << "Using uniform sampling.\n";
    
  }

  std::string used_descriptors;
  if (pcl::console::parse_argument (argc, argv, "--descriptors", used_descriptors) != -1){
    if (used_descriptors.compare ("shot") == 0)
      fpfh = false;
    else if (used_descriptors.compare ("fpfh") == 0)
      fpfh = true;
    else
    {
      std::cout << "Wrong descriptors type .\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
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
}


inline void showKeyHelp(){
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
  std::cout << "Press v to toggle verbose" << std::endl;
}


void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event){
  std::string pressed;
  pressed = event.getKeySym ();
  if(event.keyDown()){
    if(pressed == "q"){
      cg_thresh++;
      std::cout << "\tcg_thresh increased to " << cg_thresh << std::endl;
    } else if(pressed == "w"){
      cg_thresh--;
      std::cout << "\tcg_thresh decreased to " << cg_thresh << std::endl;
    } else if(pressed == "a"){
      cg_size += 0.001;
      std::cout << "\tcg_size increased to " << cg_size << std::endl;
    } else if(pressed == "s"){
      cg_size -= 0.001;
      std::cout << "\tcg_size decreased to " << cg_size << std::endl;
    } else if(pressed == "z"){
      scene_ss += 0.001;
      std::cout << "\tscene sampling size increased to " << scene_ss << std::endl;
    } else if(pressed == "x"){
      scene_ss -= 0.001;
      std::cout << "\tscene sampling size decreased to " << scene_ss << std::endl;
    } else if(pressed == "p"){
      std::cout << "Parameters:" <<std::endl;
      std::cout << "cg_thresh " << cg_thresh << std::endl;
      std::cout << "cg_size " << cg_size << std::endl;
      std::cout << "sampling size " << scene_ss << std::endl;
    } else if(pressed == "e"){
      sac_seg_distance += 0.001;
      std::cout << "\t sac segmentation distance increased to " << sac_seg_distance <<std::endl;
    } else if(pressed == "e"){
      sac_seg_distance -= 0.001;
      std::cout << "\t sac segmentation distance decreased to " << sac_seg_distance <<std::endl;
    } else if(pressed == "k"){
      show_filtered = !show_filtered;
    } else if(pressed == "n"){
      distance += 100;
    } else if(pressed == "m"){
      if(distance > 200)
        distance -= 100;
    } else if(pressed == "i"){
      use_icp = !use_icp;
    } else if(pressed == "h"){
      showKeyHelp();
    }
  }
}


inline void SetViewPoint(pcl::PointCloud<PointType>::Ptr cloud){

    cloud->sensor_origin_.setZero();
    cloud->sensor_orientation_.w () = 0.0;
    cloud->sensor_orientation_.x () = 1.0;
    cloud->sensor_orientation_.y () = 0.0;
    cloud->sensor_orientation_.z () = 0.0;
}


std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ReadModels (char** argv) {
  pcl::PCDReader reader;

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_models;

  ifstream pcd_file_list (argv[1]);
  while (!pcd_file_list.eof())
  {
    char str[512];
    pcd_file_list.getline (str, 512);
    if(std::strlen(str) > 2 ){
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      reader.read (str, *cloud);
      ///SetViewPoint(cloud);
      cloud_models.push_back (cloud);
      PCL_INFO ("Model read: %s\n", str);
    }
  }
  std::cout << "all loaded" << std::endl;
  return std::move(cloud_models);
}


void PrintTransformation(ClusterType cluster){
  for (size_t i = 0; i < std::get<0>(cluster).size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << std::get<1>(cluster)[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = std::get<0>(cluster)[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = std::get<0>(cluster)[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }
}


class ColorSampling{
public:
  ColorSampling(float tollerance_range): tollerance_range_(tollerance_range) {
    clear();
    rgb2yuv << 0.229,    0.587,   0.114,
              -0.14713, -0.28886, 0.436,
               0.615,   -0.51499, -0.10001;
  }

  void clear(){
    avg_u_  = 0, avg_v_ = 0;
  }

  void addCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud){
    clear();
    float u, v;
    for(auto point : cloud.points){
      RGBtoYUV(point, u, v);
      avg_u_ += u;
      avg_v_ += v;
    }
    avg_u_ = avg_u_ / cloud.points.size();
    avg_v_ = avg_v_ / cloud.points.size();
  }

  void filterPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &in_cloud, pcl::PointCloud<pcl::PointXYZRGB> &out_cloud){
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    int points = in_cloud.points.size();

    for(auto point : in_cloud.points){
      if(!toFilter(point))
      cloud.points.push_back(point);
    }
    out_cloud = cloud;
    std::cout << "Point number: \n\t Original point cloud: " <<points << " \n\t Filtered point cloud: " << cloud.points.size() << std::endl;
  }

  void printColorInfo(){
    std::cout << "avg U: " << avg_u_ << " V: " << avg_v_ << std::endl;
  }

  Eigen::Matrix3f rgb2yuv;

  float avg_u_, avg_v_;
  float tollerance_range_;

  bool toFilter(const pcl::PointXYZRGB &point){
    float u, v;
    RGBtoYUV(point, u, v);
    float distance = sqrt(pow(u - avg_u_, 2) + pow(v - avg_v_, 2));
    if(distance < tollerance_range_)
      return false;
    else
      return true;
  }

  void RGBtoYUV(const pcl::PointXYZRGB &point, float &u, float &v){
    Eigen::Vector3f rgb((float)point.r / 255, (float)point.g / 255, (float)point.b / 255);
    Eigen::Vector3f yuv = rgb2yuv * rgb;
    u = yuv.y();
    v = yuv.z();
  }
};


pcl::PointCloud<pcl::PointNormal>::Ptr subsampleAndCalculateNormals (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_subsampled (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::VoxelGrid<pcl::PointXYZ> subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (subsampling_leaf_size);
  subsampling_filter.filter (*cloud_subsampled);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_subsampled_normals (new pcl::PointCloud<pcl::Normal> ());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation_filter;
  normal_estimation_filter.setInputCloud (cloud_subsampled);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr search_tree (new pcl::search::KdTree<pcl::PointXYZ>);
  normal_estimation_filter.setSearchMethod (search_tree);
  normal_estimation_filter.setRadiusSearch (normal_estimation_search_radius);
  normal_estimation_filter.compute (*cloud_subsampled_normals);

  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_subsampled_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
  concatenateFields (*cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);

  PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
  return cloud_subsampled_with_normals;
}


template <class T, class Estimator>
class KeyDes{
public:
  typedef pcl::PointCloud<T> PD;
  typedef pcl::PointCloud<PointType> P;
  typedef pcl::PointCloud<NormalType> PN;
  typename PD::Ptr model_descriptors;
  typename PD::Ptr scene_descriptors;
  typename P::Ptr model ;
  typename P::Ptr model_keypoints;
  typename P::Ptr scene;
  typename P::Ptr scene_keypoints;
  typename PN::Ptr model_normals;
  typename PN::Ptr scene_normals;
  bool created;


  KeyDes(P::Ptr model, P::Ptr model_keypoints, P::Ptr scene, P::Ptr scene_keypoints, PN::Ptr model_normals, PN::Ptr scene_normals ):
            model_descriptors (new PD ()),
            scene_descriptors (new PD ()),
            model(model),
            model_keypoints(model_keypoints),
            scene(scene),
            scene_keypoints(scene_keypoints),
            model_normals(model_normals),
            scene_normals(scene_normals),
            created(false){}

  pcl::CorrespondencesPtr run()
  {

    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    //create scene descriptors
    std::cout << "calculating scene descriptors "  <<std::endl;
    Estimator est;
    est.setInputCloud(scene_keypoints);
    est.setSearchSurface(scene);
    est.setInputNormals(scene_normals);
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    est.setSearchMethod(tree);
    est.setRadiusSearch (descr_rad);
    est.compute (*scene_descriptors);

    if(!created){
      //create model descriptors
      std::cout << "calculating model descriptors "  <<std::endl;
      est.setInputCloud(model_keypoints);
      est.setSearchSurface(model);
      est.setInputNormals(model_normals);
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGB>);
      est.setSearchMethod(tree2);
      est.compute (*model_descriptors);
      created = true;
  }

    pcl::KdTreeFLANN<T> match_search;
    //std::cout <<"calculated " << model_descriptors->size() << " for the model and " << scene_descriptors->size() << " for the scene" <<std::endl;

    //  Find Model-Scene Correspondences with KdTree
    std::cout << "calculating correspondences "  <<std::endl;

    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    #pragma omp parallel for 
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
      std::vector<int> neigh_indices (1);
      std::vector<float> neigh_sqr_dists (1);
      if(match_search.point_representation_->isValid (scene_descriptors->at(i))){
        int found_neighs = match_search.nearestKSearch (scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  0.25 std add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
          pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
          #pragma omp critical
          model_scene_corrs->push_back (corr);
        }
      }
    }
    std::cout << "\tFound "  <<model_scene_corrs->size ()<< " correspondences "<< std::endl;
    return model_scene_corrs;

  }
};


class Ppfe{
public:

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointCloud<pcl::PointXYZ>::Ptr model_xyz_ ;
  pcl::ModelCoefficients::Ptr coefficients ;
  pcl::PointIndices::Ptr inliers;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_model_input;
  pcl::PointCloud<pcl::PPFSignature>::Ptr cloud_model_ppf ;
  pcl::PPFRegistration<pcl::PointNormal, pcl::PointNormal> ppf_registration;

  pcl::PPFHashMapSearch::Ptr hashmap_search;
  unsigned nr_points;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_scene_input ;
  pcl::PointCloud<pcl::PointNormal> cloud_output_subsampled;
  Eigen::Matrix4f mat;
  ClusterType cluster;
  

  Ppfe(pcl::PointCloud<PointType>::Ptr model){
            
    hashmap_search = boost::make_shared<pcl::PPFHashMapSearch>(12.0f / 180.0f * float (M_PI), 0.05f);
    cloud_model_ppf = boost::make_shared<pcl::PointCloud<pcl::PPFSignature>>();
    inliers = boost::make_shared<pcl::PointIndices> ();
    model_xyz_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    copyPointCloud(*model, *model_xyz_);
    coefficients = boost::make_shared<pcl::ModelCoefficients> ();
    cloud_scene = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (sac_seg_iter);
    extract.setNegative (true);
    ppf_registration.setSceneReferencePointSamplingRate (reg_sampling_rate); //10
    ppf_registration.setPositionClusteringThreshold (reg_clustering_threshold); //0.2f
    ppf_registration.setRotationClusteringThreshold (30.0f / 180.0f * float (M_PI));
    cloud_model_input = subsampleAndCalculateNormals (model_xyz_);
    pcl::PPFEstimation<pcl::PointNormal, pcl::PointNormal, pcl::PPFSignature> ppf_estimator;
    ppf_estimator.setInputCloud (cloud_model_input);
    ppf_estimator.setInputNormals (cloud_model_input);
    ppf_estimator.compute (*cloud_model_ppf);
    hashmap_search->setInputFeatureCloud (cloud_model_ppf);
    ppf_registration.setSearchMethod (hashmap_search);
    ppf_registration.setInputSource (cloud_model_input);
  }


  ClusterType GetCluster(pcl::PointCloud<PointType>::Ptr scene){
    seg.setDistanceThreshold (sac_seg_distance);
    copyPointCloud(*scene, *cloud_scene);
    nr_points = unsigned (cloud_scene->points.size ());
    while (cloud_scene->points.size () > 0.3 * nr_points){
      seg.setInputCloud (cloud_scene);
      seg.segment (*inliers, *coefficients);
      PCL_INFO ("Plane inliers: %u\n", inliers->indices.size ());
      if (inliers->indices.size () < max_inliers) 
        break;
      extract.setInputCloud (cloud_scene);
      extract.setIndices (inliers);
      extract.filter (*cloud_scene);
    }
    cloud_scene_input = subsampleAndCalculateNormals (cloud_scene);

    ppf_registration.setInputTarget (cloud_scene_input);
    ppf_registration.align (cloud_output_subsampled);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output_subsampled_xyz (new pcl::PointCloud<pcl::PointXYZ> ());
    //for (size_t i = 0; i < cloud_output_subsampled.points.size (); ++i)
    //  cloud_output_subsampled_xyz->points.push_back ( pcl::PointXYZ (cloud_output_subsampled.points[i].x, cloud_output_subsampled.points[i].y, cloud_output_subsampled.points[i].z));
    mat = ppf_registration.getFinalTransformation ();
    std::vector<pcl::Correspondences> cor_tmp;

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4, 0, 4, 4>>> mat_tmp;
    mat_tmp.push_back(mat);
    ClusterType cluster = std::make_tuple(mat_tmp, cor_tmp);
    inliers->indices.clear();
    return cluster;
  }

  pcl::PointCloud<PointType>::Ptr GetModelKeypoints(){
    pcl::PointCloud<PointType>::Ptr tmp(new pcl::PointCloud<PointType>());
    copyPointCloud(*cloud_model_input, *tmp);
    return tmp;
  }

  pcl::PointCloud<PointType>::Ptr GetSceneKeypoints(){
    pcl::PointCloud<PointType>::Ptr tmp(new pcl::PointCloud<PointType>());
    copyPointCloud(*cloud_scene, *tmp);
    return tmp;
  }

};


class OpenniStreamer {
public:
  openni::Device device_;        // Software object for the physical device i.e.  
  openni::VideoStream ir_;       // IR VideoStream Class Object
  openni::VideoStream color_;    // Color VideoStream Class Object
  openni::Status rc_;

  OpenniStreamer () {
    rc_ = openni::OpenNI::initialize(); // Initialize OpenNI 
    if(rc_ != openni::STATUS_OK){ 
      std::cout << "OpenNI initialization failed" << std::endl; 
      openni::OpenNI::shutdown(); 
    } 
    else 
      std::cout << "OpenNI initialization successful" << std::endl; 

    rc_ = device_.open(openni::ANY_DEVICE); 
    if(rc_ != openni::STATUS_OK){ 
      std::cout << "Device initialization failed" << std::endl; 
      device_.close(); 
    }
    rc_ = ir_.create(device_, openni::SENSOR_DEPTH);    // Create the VideoStream for IR
    
    if(rc_ != openni::STATUS_OK){
        std::cout << "Ir sensor creation failed" << std::endl;
        ir_.destroy();
    }
    else
        std::cout << "Ir sensor creation successful" << std::endl;
    rc_ = ir_.start();                      // Start the IR VideoStream
    //ir.setMirroringEnabled(TRUE); 
    if(rc_ != openni::STATUS_OK){
        std::cout << "Ir activation failed" << std::endl;
        ir_.destroy();
    }
    else
        std::cout << "Ir activation successful" << std::endl;
    device_.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

    //ir.setImageRegistrationMode(ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR);
    rc_ = color_.create(device_, openni::SENSOR_COLOR);    // Create the VideoStream for Color

    if(rc_ != openni::STATUS_OK){
        std::cout << "Color sensor creation failed" << std::endl;
        color_.destroy();
    }
    else
        std::cout << "Color sensor creation successful" << std::endl;
    rc_ = color_.start();                      // Start the Color VideoStream
     
    if(rc_ != openni::STATUS_OK){
        std::cout << "Color sensor activation failed" << std::endl;
        color_.destroy();
    }
    else
        std::cout << "Color sensor activation successful" << std::endl;
  }
};


class NormalEstimator{
public:

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;

  NormalEstimator() {
    norm_est.setKSearch (10);
  }

  NormalEstimator(int n_neighbours): NormalEstimator() { //c++11 standard
    norm_est.setKSearch (n_neighbours);
  }

  pcl::PointCloud<NormalType>::Ptr get_normals(pcl::PointCloud<PointType>::Ptr cloud){
    pcl::PointCloud<NormalType>::Ptr normals(new pcl::PointCloud<NormalType> ());
    norm_est.setInputCloud (cloud);
    norm_est.compute (*normals);
    return normals;
  }
};


class DownSampler{
public:
  pcl::VoxelGrid<pcl::PointXYZRGB> down_sampler_;

  DownSampler() {
    down_sampler_.setLeafSize (0.001, 0.001, 0.001);
  }

  DownSampler(float x, float y, float z){
    down_sampler_.setLeafSize(x, y, z);
  }

  void SetSampleSize(float x, float y, float z){
    down_sampler_.setLeafSize(x, y, z);
  }

  void DownSample(pcl::PointCloud<PointType>::Ptr cloud){
    down_sampler_.setInputCloud(cloud);
    down_sampler_.filter(*cloud);
  }
};


class Narf{
public:
  pcl::PointCloud<int> cloud_keypoint_indices_;
  Eigen::Affine3f cloud_sensor_pose_;
  bool rotation_invariant_;
  pcl::RangeImageBorderExtractor range_image_border_extractor_;
  pcl::NarfKeypoint narf_keypoint_detector_;




  Narf(): rotation_invariant_(true), cloud_sensor_pose_(Eigen::Affine3f::Identity ()) {
    narf_keypoint_detector_.setRangeImageBorderExtractor (&range_image_border_extractor_);
    narf_keypoint_detector_.getParameters().support_size = support_size;

  }

  void GetKeypoints(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints){

    boost::shared_ptr<pcl::RangeImage>cloud_range_image_ptr_(new pcl::RangeImage);

    cloud_sensor_pose_ = Eigen::Affine3f (Eigen::Translation3f (cloud->sensor_origin_[0],
                                                                cloud->sensor_origin_[1],
                                                                cloud->sensor_origin_[2])) *
                                                                Eigen::Affine3f (cloud->sensor_orientation_);

    pcl::RangeImage& cloud_range_image_ = *cloud_range_image_ptr_;

    narf_keypoint_detector_.setRangeImage (&cloud_range_image_);
    

    cloud_range_image_.createFromPointCloud(*cloud, pcl::deg2rad(0.5f), pcl::deg2rad(360.0f), pcl::deg2rad(180.0f),
                                             cloud_sensor_pose_, pcl::RangeImage::CAMERA_FRAME, 0.0, 0.0f, 1);
    
    cloud_range_image_.setUnseenToMaxRange();
    
    narf_keypoint_detector_.compute(cloud_keypoint_indices_);
    
    cloud_keypoints->points.resize (cloud_keypoint_indices_.points.size ());
    
    #pragma omp parallel for
    for (size_t i=0; i<cloud_keypoint_indices_.points.size (); ++i)
      cloud_keypoints->points[i].getVector3fMap () = cloud_range_image_.points[cloud_keypoint_indices_.points[i]].getVector3fMap ();
  }
};


class Sift{
public:
  pcl::PointCloud<pcl::PointWithScale> cloud_result_;
  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_;

  Sift():tree_(new pcl::search::KdTree<pcl::PointXYZRGB> ()){
    sift_.setSearchMethod(tree_);
    sift_.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift_.setMinimumContrast(min_contrast);
  }

  void GetKeypoints(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints) {

    sift_.setInputCloud(cloud);
    sift_.compute(cloud_result_);
    copyPointCloud(cloud_result_, *cloud_keypoints);
  }
};


template <class T>
class Ransac{
public:
  std::vector<int> cloud_inliers;

  void GetKeypoints(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints){

  typename T::Ptr cloud_plane (new T (cloud));

  pcl::RandomSampleConsensus<pcl::PointXYZRGB> model_ransac (cloud_plane);
  model_ransac.computeModel();
  model_ransac.getInliers(cloud_inliers);

  pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, cloud_inliers, *cloud_keypoints);
  }

};


class Uniform{
public:
  pcl::UniformSampling<PointType> uniform_sampling;
  pcl::PointCloud<int> sampled_indices;
  float cloud_ss_ = 0;

  void SetSamplingSize(float sampling_size){
    cloud_ss_ = sampling_size;
  }

  void GetKeypoints(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_keypoints){
    if (cloud_ss_ != 0){
      uniform_sampling.setInputCloud (cloud);
      uniform_sampling.setRadiusSearch (cloud_ss_);
      uniform_sampling.compute (sampled_indices);
      pcl::copyPointCloud (*cloud, sampled_indices.points, *cloud_keypoints);
    }else
      std::cout << "no sampling size inserted" << std::endl;
  }

};


class Hough{
public:
  ClusterType cluster;
  pcl::PointCloud<RFType>::Ptr model_rf_;
  pcl::PointCloud<RFType>::Ptr scene_rf_;
  pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est_;
  pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer_;
  bool created;
  


  Hough(): model_rf_(new pcl::PointCloud<RFType> ()), scene_rf_(new pcl::PointCloud<RFType> ()), created(false) {
    rf_est_.setFindHoles (true);
    rf_est_.setRadiusSearch (rf_rad_);
    clusterer_.setHoughBinSize (cg_size);
    clusterer_.setHoughThreshold (cg_thresh);
    clusterer_.setUseInterpolation (true);
    clusterer_.setUseDistanceWeight (false);

  }


  ClusterType GetClusters(pcl::PointCloud<PointType>::Ptr model, pcl::PointCloud<PointType>::Ptr model_keypoints,
                   pcl::PointCloud<NormalType>::Ptr model_normals, pcl::PointCloud<PointType>::Ptr scene,
                    pcl::PointCloud<PointType>::Ptr scene_keypoints, pcl::PointCloud<NormalType>::Ptr scene_normals,
                    pcl::CorrespondencesPtr model_scene_corrs){

    clusterer_.setHoughBinSize (cg_size);
    clusterer_.setHoughThreshold (cg_thresh);
    //  Compute (Keypoints) Reference Frames only for Hough
    if(!created){
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
    if(!created){
      clusterer_.setInputCloud (model_keypoints);
      clusterer_.setInputRf (model_rf_);
      created = true;
    }

    clusterer_.setSceneCloud (scene_keypoints);
    clusterer_.setSceneRf (scene_rf_);
    clusterer_.setModelSceneCorrespondences (model_scene_corrs);

    //std::cout << "prepared Hough for clustering" <<std::endl;

    //clusterer_.cluster (clustered_corrs);
    clusterer_.recognize (std::get<0>(cluster), std::get<1>(cluster));
    return cluster;
  }

};


class GCG{
public:
  pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer_;
  ClusterType cluster;

  GCG(){
    gc_clusterer_.setGCSize (cg_size);
    gc_clusterer_.setGCThreshold (cg_thresh);
  }


  ClusterType GetClusters(pcl::PointCloud<PointType>::Ptr model_keypoints, pcl::PointCloud<PointType>::Ptr scene_keypoints, pcl::CorrespondencesPtr model_scene_corrs){

    gc_clusterer_.setInputCloud (model_keypoints);
    gc_clusterer_.setSceneCloud (scene_keypoints);
    gc_clusterer_.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer_.cluster (clustered_corrs);
    gc_clusterer_.recognize (std::get<0>(cluster), std::get<1>(cluster));
    return cluster;

  }
};


template<class T, class TT>
class ICPRegistration{
public:
  int count;
  pcl::IterativeClosestPoint<T, TT> icp;


  ICPRegistration(){
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance (0.05);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (20);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-18);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon (1);
    count = 0;
  }

  void Align(typename pcl::PointCloud<T>::Ptr cloud_source, typename pcl::PointCloud< TT>::Ptr cloud_target){
    
    typename pcl::PointCloud< T>::Ptr cloud_source_registered (new typename pcl::PointCloud< T> ());
    
    icp.setInputSource (cloud_source);
    icp.setInputTarget (cloud_target);

    // Perform the alignment
    icp.align (*cloud_source);

    // Obtain the transformation that aligned cloud_source to cloud_source_registered
    //transformation = icp.getFinalTransformation ();

    //std::cout << "TRANSFORMATION: " << std::endl;
    //std::cout << transformation << std::endl;
  } 
};


class Visualizer{
public:
  pcl::visualization::PCLVisualizer viewer_;
  pcl::PointCloud<PointType>::Ptr off_scene_model_;
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints_;
  ICPRegistration<PointType, PointType> icp;
  int iter;
  bool clean;
  std::stringstream ss_cloud;
  std::stringstream ss_line;
  std::vector<std::string> to_remove;

  Visualizer(): off_scene_model_(new pcl::PointCloud<PointType> ()), off_scene_model_keypoints_(new pcl::PointCloud<PointType> ()), iter(0), clean(true) {
      viewer_.registerKeyboardCallback (keyboardEventOccurred);

  }

  void Visualize(pcl::PointCloud<PointType>::Ptr model, pcl::PointCloud<PointType>::Ptr model_keypoints, pcl::PointCloud<PointType>::Ptr scene,
                 pcl::PointCloud<PointType>::Ptr scene_keypoints, ClusterType cluster, pcl::PointCloud<PointType>::Ptr filtered_scene){


    if(!clean) {
     for (auto s : to_remove)
       viewer_.removeShape(s);
      clean = true;
      to_remove.clear();
    }
    //SetViewPoint(scene);
    pcl::transformPointCloud (*model, *off_scene_model_, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    if(show_filtered){
      scene = filtered_scene;
    }
    if(iter == 0){
      viewer_.addPointCloud (off_scene_model_,  "off_scene_model_");
      viewer_.addPointCloud (scene, "scene_cloud");
    }
    else
      viewer_.updatePointCloud (scene, "scene_cloud");

    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints_, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    
    if (show_keypoints){
      pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
      SetViewPoint(scene_keypoints);
      if(iter == 0)
        viewer_.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
      else
        viewer_.updatePointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
      viewer_.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

      pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints__color_handler (off_scene_model_keypoints_, 0, 0, 255);
      SetViewPoint(off_scene_model_keypoints_);
      if(iter == 0)
        viewer_.addPointCloud (off_scene_model_keypoints_, off_scene_model_keypoints__color_handler, "off_scene_model_keypoints_");
      else
        viewer_.updatePointCloud (off_scene_model_keypoints_, off_scene_model_keypoints__color_handler, "off_scene_model_keypoints_");
      viewer_.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints_");
    }

    for (size_t i = 0; i < std::get<0>(cluster).size (); ++i){
      clean = false;
      pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
      pcl::transformPointCloud (*model, *rotated_model, std::get<0>(cluster)[i]);
      if(use_icp){
        icp.Align(rotated_model, filtered_scene);
        //pcl::transformPointCloud (*rotated_model, *rotated_model, transformation);
      }
      SetViewPoint(rotated_model);

      ss_cloud << "instance" << i;
      to_remove.push_back(ss_cloud.str());

      pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);

      viewer_.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

      //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
      if (show_correspondences){
        for (size_t j = 0; j < std::get<1>(cluster)[i].size(); ++j){
          ss_line << "correspondence_line" << i << "_" << j << "_" << iter;
          float model_x = off_scene_model_keypoints_->at (std::get<1>(cluster)[i][j].index_query).x;
          float model_y = off_scene_model_keypoints_->at (std::get<1>(cluster)[i][j].index_query).y;
          float model_z = off_scene_model_keypoints_->at (std::get<1>(cluster)[i][j].index_query).z;
          float scene_x = scene_keypoints->at (std::get<1>(cluster)[i][j].index_match).x;
          float scene_y = scene_keypoints->at (std::get<1>(cluster)[i][j].index_match).y;
          float scene_z = scene_keypoints->at (std::get<1>(cluster)[i][j].index_match).z;

          Eigen::Quaternion<float> transformation(0, 1, 0, 0);
          Eigen::Vector3f tmp(model_x, model_y, model_z);
          tmp = transformation._transformVector(tmp);
          pcl::PointXYZ model_point(tmp.x(), tmp.y(), tmp.z());

          Eigen::Vector3f tmp2(scene_x, scene_y, scene_z);
          tmp2 = transformation._transformVector(tmp2);
          pcl::PointXYZ scene_point(tmp2.x(), tmp2.y(), tmp2.z());

          viewer_.addLine<pcl::PointXYZ, pcl::PointXYZ> (model_point, scene_point, 0, 255, 0, ss_line.str ());
          to_remove.push_back(ss_line.str());
        }
      }
    }
  viewer_.spinOnce();
  iter++;
  }
};


