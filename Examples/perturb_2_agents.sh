#!/bin/bash
dataset="EuRoC"
BAG_PATH="/home/lidonghao/MultiAgent/data/EuRoC" # Change this to your dataset directory
WS_PATH="/home/lidonghao/ws/covins_ws/src"  # Change this to your workspace directory
TEST_AGENTS=("MH_01_easy" "MH_02_easy")
Perturb_agent=0
# METHODS=(""gaussian_blur" "defocus_blur" "fog" "frost" 
# "contrast" "brightness" "snow" "spatter")
output_path="${WS_PATH}/covins/covins_backend/output/" # This is default output path for COVINS-G, change it if you have different output path
frequency=500
duration=100

Experiment="dncnn_gray_de_f${frequency}_p${duration}"
perturb_result="${WS_PATH}/ablation_perturbation/results/EuRoC/agn0/perturbed_results/$Experiment/" # Change this to your perturbation result directory
# METHODS=("fog")
# severity=2
# BMETHODS=("gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "contrast" "pixelate" "jpeg_compression")
BMETHODS=("gaussian_noise" )
# "${Experiment},${method},${severity},$i,${mean_KF0},${std_KF0},${rmse_KF0},${mean_KF1},${std_KF1},${rmse_KF1}"
denoise=0

overall_log_path="${perturb_result}experiment_log.csv"
if [ ! -d ${perturb_result} ]; then
    echo "No directory found, ending the script"
    exit 1
fi
echo "Experiment,method,severity,trial,ag0_mean,ag0_std,ag0_rmse,ag1_mean,ag1_std,ag1_rmse" >> ${overall_log_path}

for method in ${BMETHODS[*]};
do
    for severity in {1..5..4};
    do  
        for i in {1..5};
        do 
            echo " ===============severity: ${severity} ==================="
            if [ $denoise -eq 1 ]; then
                echo " ++++++++++++++ denoise +++++++++++++++++"
                trail_name="denoised_${method}_n${Perturb_agent}_lv${severity}_exp1/"
            else
                echo " ++++++++++++++ simple perturb +++++++++++++++++"
                trail_name="${method}_n${Perturb_agent}_lv${severity}_exp1/"
            fi
            out_dir="${perturb_result}${trail_name}"

            

            # echo "Experiment,method,severity,trial,ag0_mean,ag0_std,ag0_rmse,ag1_mean,ag1_std,ag1_rmse" >> ${overall_log_path}


            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "Starting ROSCORE"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            roscore >/dev/null 2>&1 & ROSCORE_PID=$!
            sleep 2

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "Starting perturbation ${method} with severity ${severity} trial ${i}"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            # Initialize visualization
            roslaunch ~/ws/covins_ws/src/covins/covins_backend/launch/tf.launch >/dev/null 2>&1 & TF_PID=$!
            rviz -d ~/ws/covins_ws/src/covins/covins_backend/config/covins.rviz >/dev/null 2>&1 & RVIZ_PID=$!

            # Launch COVINS-G backend
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "COVINS-G backend running"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            rosrun covins_backend covins_backend_node & COVINS_PID=$!

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "waiting 20s for COVINS-G connection" & sleep 20
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

            # Launch ORB_SLAM front-end
            echo "ORB_SLAM agent 0 running" & roslaunch ORB_SLAM3 launch_ros_euroc.launch perturbed:=true & ORB_PID0=$!
            echo "ORB_SLAM agent 1 running" & roslaunch ORB_SLAM3 launch_ros_euroc.launch perturbed:=false ag_n:=1 & ORB_PID1=$!

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "waiting 20s for ORB_SLAM connection" & sleep 20
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "agents bags playing" 

            if [ $Perturb_agent -eq 0 ]; then
                Perturbed_bag="${out_dir}perturbed_images.bag"
                echo "agen0 playing perturbed" & rosbag play $Perturbed_bag --start 45 & BAG_PID0=$!
            else
                echo "agen0 playing normal" & rosbag play $BAG_PATH/${TEST_AGENTS[0]}.bag /cam0/image_raw:=/cam0/image_raw0 /cam1/image_raw:=/cam1/image_raw0 /imu0:=/imu0 --start 45 & BAG_PID0=$!
            fi
            # rosrun ablation_perturbation  rosbag_perturbation.py --agent_number $Perturb_agent --noise_type ${method} --severity ${severity} --trail ${i} & PERTURB_PID=$!

            echo "agen1 playing" & rosbag play $BAG_PATH/${TEST_AGENTS[1]}.bag /cam0/image_raw:=/cam0/image_raw1 /cam1/image_raw:=/cam1/image_raw1 /imu0:=/imu1 --start 35 & BAG_PID1=$!

            wait $BAG_PID0 $BAG_PID1


            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "Everything is done, killing the processes"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            # kill $ORB_PID0 # $ORB_PID1

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "waiting 30s for backend compute" & sleep 30
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

            kill $ORB_PID0 $ORB_PID1


            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            log_dir=$out_dir
            echo "Saving logs to ${log_dir}"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        
            mkdir -p ${log_dir}/KF_0
            mkdir -p ${log_dir}/KF_1
            mv ${output_path}KF_0_ftum.csv ${log_dir}/KF_0_ftum.csv
            mv ${output_path}KF_1_ftum.csv ${log_dir}/KF_1_ftum.csv
            mv ${output_path}stamped_traj_estimate.txt ${log_dir}/stamped_traj_estimate.txt

            mean_KF0=$(grep -oE 'mean\s+\S+' ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')
            std_KF0=$(grep -oE 'std\s+\S+' ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')
            rmse_KF0=$(grep -oE 'rmse\s+\S+' ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')


            mean_KF1=$(grep -oE 'mean\s+\S+' ${log_dir}KF_1/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.txt | grep -oE '\S+$')
            std_KF1=$(grep -oE 'std\s+\S+' ${log_dir}KF_1/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.txt | grep -oE '\S+$')
            rmse_KF1=$(grep -oE 'rmse\s+\S+' ${log_dir}KF_1/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.txt | grep -oE '\S+$')

            echo "${Experiment},${method},${severity},$i,${mean_KF0},${std_KF0},${rmse_KF0},${mean_KF1},${std_KF1},${rmse_KF1}" >> ${overall_log_path}
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "Finished perturbation ${method} with severity ${severity} trial ${i}"
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

            kill $COVINS_PID

            for pid in $(ps -ef | grep -e "ros" -e "rviz" | awk '{print $2}'); do kill -9 $pid; done

            echo "All processes killed and cleaned up!" & sleep 10
        done
        evo_ape euroc ${output_path}MH_01_easy/mav0/state_groundtruth_estimate0/data.csv ${log_dir}KF_0_ftum.csv -vas \
                --save_plot ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.png \
                --save_results ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.zip \
                --logfile ${log_dir}KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt \
                --t_max_diff 1.0

        evo_ape euroc ${output_path}MH_02_easy/mav0/state_groundtruth_estimate0/data.csv ${log_dir}KF_1_ftum.csv -vas \
                --save_plot ${log_dir}KF_1/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.png \
                --save_results ${log_dir}/KF_1MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.zip \
                --logfile ${log_dir}KF_1/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_1.txt \
                --t_max_diff 1.0

    # echo "${Experiment},${method},${severity},,,,,,," >> ${overall_log_path}
    done

# echo "${Experiment},${method},,,,,,,," >> ${overall_log_path}
done





