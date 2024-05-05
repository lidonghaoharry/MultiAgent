#!/bin/bash
dataset="EuRoC"
BAG_PATH="/home/lidonghao/MultiAgent/data/EuRoC"
TEST_AGENTS=("MH_01_easy" "MH_02_easy")
Perturb_agent=0
# METHODS=(""gaussian_blur" "defocus_blur" "fog" "frost" 
# "contrast" "brightness" "snow" "spatter")
output_path="/home/lidonghao/ws/covins_ws/src/covins/covins_backend/output/"
frequency=500
duration=100
Experiment="jpeg_compression_f${frequency}_p${duration}"
perturb_result="/home/lidonghao/ws/covins_ws/src/ablation_perturbation/results/EuRoC/agn0/perturbed_results/$Experiment/"
overall_log_path="${output_path}${Experiment}/experiment_log.csv"
# severity=2
# BMETHODS=("gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "contrast" "pixelate" "jpeg_compression")
METHODS=("jpeg_compression")
# if [ ! -d ${output_path}${Experiment} ]; then
#     mkdir -p ${output_path}${Experiment}
# fi

# echo "Experiment,agent,method,severity,mean,std,rmse" >> ${overall_log_path}
denoise=1

for method in ${METHODS[*]};
do
    for severity in {1..5..4};
    do
        i=1

        echo " ===============severity: ${severity} ==================="
        if [ $denoise -eq 1 ]; then
            echo " ++++++++++++++ denoise +++++++++++++++++"
            trail_name="denoised_${method}_n${Perturb_agent}_lv${severity}_exp${i}/"
        else
            echo " ++++++++++++++ simple perturb +++++++++++++++++"
            trail_name="${method}_n${Perturb_agent}_lv${severity}_exp${i}/"
        fi
        out_dir="${perturb_result}${trail_name}"
        echo " ++++++++++++++ SAVING TO: out_dir: ${out_dir} +++++++++++++++++"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Starting ROSCORE"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        roscore >/dev/null 2>&1 & ROSCORE_PID=$!
        sleep 2

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Starting perturbation ${method} with severity ${severity}"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        # Initialize visualization
        roslaunch ~/ws/covins_ws/src/covins/covins_backend/launch/tf.launch >/dev/null 2>&1 & TF_PID=$!
        rviz -d ~/ws/covins_ws/src/covins/covins_backend/config/covins.rviz >/dev/null 2>&1  & RVIZ_PID=$!

        # Launch COVINS-G backend
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "COVINS-G backend running"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        rosrun covins_backend covins_backend_node & COVINS_PID=$!

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "waiting 20s for COVINS-G connection" & sleep 20
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        # Launch ORB_SLAM front-end
        {   if [ $severity -eq 0 ]; then
                echo "ORB_SLAM NOT PERTURBED, log_path: $out_dir"
                echo "ORB_SLAM agent 0 running" & roslaunch ORB_SLAM3 launch_ros_euroc.launch perturbed:=false log_path:=$out_dir & ORB_PID0=$!
            else
                echo "ORB_SLAM PERTURBED, log_path: $out_dir"
                echo "ORB_SLAM agent 0 running" & roslaunch ORB_SLAM3 launch_ros_euroc.launch perturbed:=true log_path:=$out_dir & ORB_PID0=$!
            fi
            

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "waiting 20s for ORB_SLAM connection" & sleep 20
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            if [ $severity -eq 0 ]; then
                echo "NOT PERTURBING"
                echo "agen0 playing normal" & rosbag play $BAG_PATH/${TEST_AGENTS[0]}.bag /cam0/image_raw:=/cam0/image_raw0 /cam1/image_raw:=/cam1/image_raw0 /imu0:=/imu0 --start 45 & BAG_PID0=$!
                
            else
                Perturbed_bag="${out_dir}perturbed_images.bag"
                echo "agen0 playing perturbed" & rosbag play $Perturbed_bag --start 45 & BAG_PID0=$!
            fi
            

            wait $BAG_PID0

            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            echo "waiting 30s for backend compute" & sleep 30
            echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

            kill $ORB_PID0
        }

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        log_dir="${output_path}${Experiment}/${method}_s${severity}_trial${i}/"
        echo "Saving logs to ${log_dir}"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        
        # mkdir -p ${log_dir}/KF_0
        # mkdir -p ${log_dir}/KF_1

        # mv ${output_path}KF_0_ftum.csv ${log_dir}/KF_0_ftum.csv
        # mv ${output_path}KF_1_ftum.csv ${log_dir}/KF_1_ftum.csv
        # mv ${output_path}stamped_traj_estimate.txt ${log_dir}/stamped_traj_estimate.txt
        # evo_ape euroc ${output_path}MH_01_easy/mav0/state_groundtruth_estimate0/data.csv ${log_dir}/KF_0_ftum.csv -vas \
        #         --save_plot ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.png \
        #         --save_results ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.zip \
        #         --logfile ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt \
        #         --t_max_diff 1.0

        # mean_KF0=$(grep -oE 'mean\s+\S+' ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')
        # std_KF0=$(grep -oE 'std\s+\S+' ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')
        # rmse_KF0=$(grep -oE 'rmse\s+\S+' ${log_dir}/KF_0/MH_12_p${Perturb_agent}_${method}_s${severity}_KF_0.txt | grep -oE '\S+$')

        # echo "${Experiment},KF_0,${method},${severity},${mean_KF0},${std_KF0},${rmse_KF0}" >> ${overall_log_path}

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Finished perturbation ${method} with severity ${severity}"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        

        kill $COVINS_PID

        for pid in $(ps -ef | grep -e "ros" -e "rviz" | awk '{print $2}'); do kill -9 $pid; done

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Plot keypoints number trend"
        python /home/lidonghao/ws/covins_ws/src/ablation_perturbation/src/plot_feature_trend.py --exp ${Experiment} --trail ${trail_name} & PLOT_PID=$!
        wait $PLOT_PID
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        echo "All processes killed and cleaned up!" & sleep 10
    done


done