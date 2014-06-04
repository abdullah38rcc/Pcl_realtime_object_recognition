#include "all.hpp"
#include "pcl_object_recognition.hpp"


int main( int argc, char** argv )
{

  parseCommandLine(argc, argv);
 

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr complete_scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  
  openni::Device device;        // Software object for the physical device i.e.  
  openni::VideoStream ir;       // IR VideoStream Class Object
  openni::VideoStream color;    // Color VideoStream Class Object
  openni::VideoFrameRef irf; 
  openni::VideoFrameRef colorf; 

  openni::Status rc;
  rc = openni::OpenNI::initialize(); // Initialize OpenNI 
  if(rc != openni::STATUS_OK){ 
    std::cout << "OpenNI initialization failed" << std::endl; 
    openni::OpenNI::shutdown(); 
  } 
  else 
    std::cout << "OpenNI initialization successful" << std::endl; 

  rc = device.open(openni::ANY_DEVICE); 
  if(rc != openni::STATUS_OK){ 
    std::cout << "Device initialization failed" << std::endl; 
    device.close(); 
  }
  rc = ir.create(device, openni::SENSOR_DEPTH);    // Create the VideoStream for IR
  
  if(rc != openni::STATUS_OK){
      std::cout << "Ir sensor creation failed" << std::endl;
      ir.destroy();
  }
  else
      std::cout << "Ir sensor creation successful" << std::endl;
  rc = ir.start();                      // Start the IR VideoStream
  //ir.setMirroringEnabled(TRUE); 
  if(rc != openni::STATUS_OK){
      std::cout << "Ir activation failed" << std::endl;
      ir.destroy();
  }
  else
      std::cout << "Ir activation successful" << std::endl;
  device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

  //ir.setImageRegistrationMode(ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR);
  rc = color.create(device, openni::SENSOR_COLOR);    // Create the VideoStream for Color

  if(rc != openni::STATUS_OK){
      std::cout << "Color sensor creation failed" << std::endl;
      color.destroy();
  }
  else
      std::cout << "Color sensor creation successful" << std::endl;
  rc = color.start();                      // Start the Color VideoStream
   
  if(rc != openni::STATUS_OK){
      std::cout << "Color sensor activation failed" << std::endl;
      color.destroy();
  }
  else
      std::cout << "Color sensor activation successful" << std::endl;


  int save_cloud = 0;


  //load and filter model and scene
  if (pcl::io::loadPCDFile (argv[1], *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    return (-1);
  }  
   /*
  
  //pcl::removeNaNFromPointCloud(*model,*model, indexes);


  DownSampler sampler;
  sampler.DownSample(model);
  std::cout << "model dimension after sampling "  <<model->points.size() <<std::endl;

  std::cout << "cloud scene dimensions " << scene->height<< " x " << scene->width << " number of points " << scene->points.size() <<std::endl;
  if(save_cloud == 0)
    pcl::io::savePCDFileASCII ("scene.pcd", *scene);

  //pcl::removeNaNFromPointCloud(*scene,*scene, indexes2);
  sampler.DownSample(model);
  std::cout << "cloud scene dimension after sampling "  <<scene->points.size() <<std::endl;
  if(save_cloud == 0)
    pcl::io::savePCDFileASCII ("scene_sampled.pcd", *scene);

*/

  

  
  Openni2pcl<pcl::PointXYZRGB> grabber;
  NormalEstimator norm;
  ClusterType cluster;
  Uniform uniform;
  Narf narf_estimator;
  Sift sift_estimator;
  Ransac<pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>> ransac_estimator;
  
  std::cout << "calculating model normals... "  <<std::endl;
  model_normals = norm.get_normals(model);
  if(save_cloud == 0)
    pcl::io::savePCDFileASCII ("model_sampled_normals.pcd", *model_normals);
  std::cout << "model size " << model->points.size() << std::endl;
  uniform.SetSamplingSize(model_ss);
  uniform.GetKeypoints(model, model_keypoints);
  SetViewPoint(model);


    //  Visualization
  Visualizer visualizer;
  while (!visualizer.to_stop)
  {

    //grab a frame and create the pointcloud
    rc = ir.readFrame(&irf);
    rc = color.readFrame(&colorf);
    scene = grabber.get_point_cloud_openni2(colorf, irf, distance, true);
    if(save_cloud == 0)
      pcl::io::savePCDFileASCII ("good_scene.pcd", *scene);
    SetViewPoint(scene);
    //  Compute Normals

    
    std::cout << "calculating scene normals... "  <<std::endl;
    scene_normals = norm.get_normals(scene);
    if(save_cloud == 0)
      pcl::io::savePCDFileASCII ("scene_sampled_normals.pcd", *scene_normals);

    if(narf){
      //NARF
      std::cout << "finding narf keypoints..."<< std::endl;
      narf_estimator.GetKeypoints(scene, scene_keypoints);
    
    }else if(sift){
      //SIFT
      std::cout << "finding sift keypoints..."<< std::endl;
      sift_estimator.GetKeypoints(scene, scene_keypoints);

    } else if(ransac){
      //RANSAC
      std::cout << "finding ransac keypoints..."<< std::endl;
      ransac_estimator.GetKeypoints(scene, scene_keypoints);

    }else{
      //UNIFORM
      std::cout << "finding uniform sampled keypoints..."<< std::endl;
      uniform.SetSamplingSize(scene_ss);
      uniform.GetKeypoints(scene, scene_keypoints);
    }

    std::cout << "\tfound " << scene_keypoints->points.size() << " keypoints in the scene and " << model_keypoints->points.size() << " in the model"<<std::endl;

    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
    if(fpfh) {
      KeyDes<pcl::FPFHSignature33, pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> > est(model, model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
      model_scene_corrs = est.run();
    }else{
      KeyDes<pcl::SHOT352, pcl::SHOTEstimationOMP<PointType, NormalType, pcl::SHOT352> > est(model, model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
      model_scene_corrs = est.run();
    }

    std::cout << "Starting to cluster..." <<std::endl;
    if (use_hough){
      //Hough3D
      Hough hough;
      cluster = hough.GetClusters(model, model_keypoints, model_normals, scene, scene_keypoints, scene_normals, model_scene_corrs);
    }else {
      //GEOMETRIC CONSISTENCY
      GCG gcg;
      cluster = gcg.GetClusters(model, scene, model_scene_corrs);
    }

    std::cout << "\tFound " << std::get<0>(cluster).size () << " model instance/instances " <<  std::endl;
    visualizer.Visualize(model, model_keypoints, scene, scene_keypoints, cluster);

  }
  return 0;
}
