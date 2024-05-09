# This repo serves as Multi-Agent SLAM Robustness Evaluation Framwork， in ROS package format

## File structure in: 

```bash
# main folder contains all functions implemented 
Ablation_Perturbation
    |- Examples                # Contains all bash files needed for running the evaluation conveniently
        |- purturb_2_agents.sh # For running two agents synchronized, apply perturbation as needed
        |- purburb_bag.sh      # Perturbing the ROS bag and store it in with format bag or images
        |- proc_traj.sh        # Post process output trajectories
        |- test_perturb.sh     # Simple test bash for test single agent perturbation
    |- include                 # Used for cpp files as needed (no cpp implementation here)
    |- launch
        |- ablation_perturbation_agent0.launch # Example launch file for perturbation (Not completed)
    |- src 
        |- models              # Network models for image repair
        |- utils               # Utils for image repair tools
        |- image_denoise.py    # Test image denoise algorithms effectiveness
        |- perturb_utils.py    # perturbation utils
        |- plot_feature_trend.py # Plot number of features recognized by ORB-SLAM for evaluation
        |- robustness.py       # Noise types
        |- rosbag_perturbation.py # Main file used for image perturbation
    |- CMakeLists.txt          # ROS package build
    |- package.xml             # ROS package build
    |- post_proc.py            # Compare two experiments
    |- test.yaml               # test configs
```
## If you already have ROS workspace, just pull out only the packages (default: base_local_planner, jackal_gps_navigation) and cooresponding launch and config files, and put them into your workspace and build with `catkin_make`. <br />
## To start:  
ROS should be initialized already as Jackal boots up <br />

1. Connect to NovAtel and make sure get GPS-RTK (you can use software in Downloads\NovAtelApplicationSuite_1-15-0_Linux\NovAtelApplicationSuite_64bit.AppImage) <br />
2. Launch novatel ROS driver <br />

```bash
roslaunch novatel_oem7_driver oem7_tty.launch
```

3. Launch navsat_transfer_node to fuse GPS into EKF Localization

```bash
roslaunch robot_localization navsat_transform_template.launch
```

4. Edit or double-check navigation configurations in `path_follow_params.yaml` <br />

PathFollower:<br />
  - pre_traj_frame: Frame of pre-defined trajectory # (utm or odom)<br />
  - pre_traj: Predefined trajectory in pre_traj_frame # ex: [[x1, y1, w1], [x2, y2, w2],[x3, y3, w3]]<br />
  - duration: Sim time for local planner # ex: 2.0<br />
  - waypoint_factor: How many waypoints that UGV  will go in next 1 second # ex: 2<br />
  - time_spaced: True if send time spaced waypoints, otherwise send distance spaced waypoints<br />
output_dir: Output directory for logging sent plans, if None, no logging<br />

5. Start navigating <br />

```bash
cp ~/cpr_noetic_ws
source ./devel/setup.bash 
roslaunch jackal_gps_navigation path_follower.launch

```

## Tests left to be done *** :  
These are not tested and ran after implementation because I couldn't get RTK correction on GPS. There must be bugs, please fix them. 
1. Check if `GPS time` is calculated correctly and updated all the time <br />
2. Have not tested the `UTM` <-> `odom` transform yet <br />
3. Need to check if `current velocity` is updated frequently enough <br />
4. `UTM`  <-> `GPS` (if function pyproj.Proj() works) <br />
5. Not sure if transformation between `UTM`  <->  `odom` provided by navsat_transfer_node is correct or available all the time <br />
6. Need to test the accuracy of calculation of `UTM` <-> `odom` <br />
