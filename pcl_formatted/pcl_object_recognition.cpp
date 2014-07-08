
#include "all.hpp"
#include "pcl_object_recognition.hpp"



int main (int argc, char** argv ){

  parseCommandLine (argc, argv);
 

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr complete_scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());

  openni::VideoFrameRef irf; 
  openni::VideoFrameRef colorf;
  OpenniStreamer openni_streamer; 
  
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> model_list = ReadModels (argv);

  if (model_list.size () == 0){
    std::cout << " no models loaded " << std::endl;
    return 1;
  }
   
  Openni2pcl<pcl::PointXYZRGB> grabber;
  NormalEstimator norm;
  ClusterType cluster;
  Uniform uniform;
  pcl::RandomSample<pcl::PointXYZRGB> random;
  Narf narf_estimator;
  Sift sift_estimator;
  Harris harris_estimator;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;

  
  Ransac<pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>> ransac_estimator;
  Ppfe ppfe_estimator (model_list[0]);
  ColorSampling filter;

  
  if(to_filter)
  {
    filter.addCloud (*model_list[0]);
  }
  if(remove_outliers)
  {
    sor.setMeanK (50);
    sor.setStddevMulThresh (0.5);
  }

  std::cout << "calculating model normals... "  <<std::endl;
  if(!ppfe)
  {
    model_normals = norm.get_normals (model_list[0]);
    std::cout << "model size " << model_list[0]->points.size() << std::endl;
    if (random_points)
    {
      random.setInputCloud (model_list[0]);
      random.setSeed (std::rand ());
      random.setSample ( random_model_samples );
      random.filter (*model_keypoints); 
    }else
    {
      uniform.SetSamplingSize (model_ss);
      uniform.GetKeypoints (model_list[0], model_keypoints);
    }
  }else
  {
    model_keypoints = ppfe_estimator.GetModelKeypoints ();
  }


    //  Visualization
  Visualizer visualizer;
  while (!visualizer.viewer_.wasStopped ())
  {

    //grab a frame and create the pointcloud
    openni_streamer.rc_ = openni_streamer.ir_.readFrame (&irf);
    openni_streamer.rc_ = openni_streamer.color_.readFrame (&colorf);
    scene = grabber.get_point_cloud_openni2 (colorf, irf, distance, true);
    

    copyPointCloud (*scene, *complete_scene);
    if(segment)
      scene = findAndSubtractPlane (scene, segmentation_threshold, segmentation_iterations);
    if(to_filter)
    {
      filter.filterPointCloud (*scene, *scene);
    }
    if(remove_outliers)
    {
      sor.setInputCloud (scene);
      sor.filter (*scene);
    }
    
    //  Compute Normals
    std::cout << "calculating scene normals... "  <<std::endl;
    scene_normals = norm.get_normals (scene);

    if(ppfe)
    {
      //PPFEESTIMATOR
      cluster = ppfe_estimator.GetCluster (scene);
      scene_keypoints = ppfe_estimator.GetSceneKeypoints ();
      show_correspondences = false;

    }else{
      if(narf)
      {
        //NARF
        std::cout << "finding narf keypoints..."<< std::endl;
        narf_estimator.GetKeypoints (scene, scene_keypoints);
      
      }else if(sift)
      {
        //SIFT
        std::cout << "finding sift keypoints..."<< std::endl;
        sift_estimator.GetKeypoints (scene, scene_keypoints);

      } else if(ransac)
      {
        //RANSAC
        std::cout << "finding ransac keypoints..."<< std::endl;
        ransac_estimator.GetKeypoints (scene, scene_keypoints);

      } else if(harris)
      {
        //HARRIS
        std::cout << "finding harris keypoints..."<< std::endl;
        harris_estimator.GetKeypoints (scene, scene_keypoints);

      } else if(random_points)
      {
        //RANDOM
        std::cout << "using random keypoints..."<< std::endl;
        random.setInputCloud (scene);
        random.setSample ( random_scene_samples );
        random.filter (*scene_keypoints); 

      }else
      {
        //UNIFORM
        std::cout << "finding uniform sampled keypoints..."<< std::endl;
        uniform.SetSamplingSize (scene_ss);
        uniform.GetKeypoints (scene, scene_keypoints);
      }

      std::cout << "\tfound " << scene_keypoints->points.size () << " keypoints in the scene and " << model_keypoints->points.size () << " in the model"<<std::endl;

      pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
      if(fpfh) 
      {
        std::cout << "using fpfh descriptors"<< std::endl;
        KeyDes<pcl::FPFHSignature33, pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }else if(pfh) 
      {
        std::cout << "using pfh descriptors"<< std::endl;
        KeyDes<pcl::PFHSignature125, pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }else if(pfhrgb) 
      {
        std::cout << "using pfhrgb descriptors"<< std::endl;
        KeyDes<pcl::PFHRGBSignature250, pcl::PFHRGBEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHRGBSignature250> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }else if(ppf) 
      {
        std::cout << "using ppf descriptors"<< std::endl;
        KeyDes<pcl::PPFSignature, pcl::PPFEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PPFSignature> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }else if(ppfrgb) 
      {
        std::cout << "using ppfrgb descriptors"<< std::endl;
        KeyDes<pcl::PPFRGBSignature, pcl::PPFRGBEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PPFRGBSignature> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }else if(shot)
      {
        std::cout << "using shot descriptors"<< std::endl;
        KeyDes<pcl::SHOT352, pcl::SHOTEstimationOMP<PointType, NormalType, pcl::SHOT352> > est (model_list[0], model_keypoints, scene, scene_keypoints, model_normals, scene_normals);
        model_scene_corrs = est.run ();
      }

      std::cout << "Starting to cluster..." <<std::endl;
      if (use_hough)
      {
        //Hough3D
        Hough hough;
        cluster = hough.GetClusters (model_list[0], model_keypoints, model_normals, scene, scene_keypoints, scene_normals, model_scene_corrs);
      }else 
      {
        //GEOMETRIC CONSISTENCY
        GCG gcg;
        cluster = gcg.GetClusters (model_list[0], scene, model_scene_corrs);
      }
    }
    std::cout << "\tFound " << std::get<0>(cluster).size () << " model instance/instances " <<  std::endl;
    
    SetViewPoint (complete_scene);

    visualizer.Visualize (model_list[0], model_keypoints, complete_scene, scene_keypoints, cluster, scene);
  }
  return 0;
}
