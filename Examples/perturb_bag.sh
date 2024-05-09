#!/bin/bash
dataset="EuRoC"
BAG_PATH="/home/lidonghao/MultiAgent/data/EuRoC" # Change this to your dataset directory
WS_PATH="/home/lidonghao/ws/covins_ws/src"  # Change this to your workspace directory
output_path="${WS_PATH}/covins/covins_backend/output/" # This is default output path for COVINS-G, change it if you have different output path
TEST_AGENTS=("MH_01_easy" "MH_02_easy")
Perturb_agent=0
# METHODS=(""gaussian_blur" "defocus_blur" "fog" "frost" 
# "contrast" "brightness" "snow" "spatter")
frequency=500
duration=100
Experiment="jpeg_compression_f${frequency}_p${duration}"
perturb_result="${WS_PATH}/ablation_perturbation/results/EuRoC/agn0/perturbed_results/$Experiment/"
METHODS=("jpeg_compression")
denoise=0
# severity=2
# BMETHODS=("gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "contrast" "pixelate" "jpeg_compression")
# echo "Experiment,agent,method,severity,mean,std,rmse" >> ${overall_log_path}

for method in ${METHODS[*]};
do
    for severity in {1..5..4};
    do
        i=1
        if [ $denoise -eq 1 ]; then
            out_dir="denoised_${perturb_result}${method}_n${Perturb_agent}_lv${severity}_exp${i}/"
        else
            out_dir="${perturb_result}${method}_n${Perturb_agent}_lv${severity}_exp${i}/"
        fi
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Starting ROSCORE"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        roscore >/dev/null 2>&1 & ROSCORE_PID=$!
        sleep 2

        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "Starting perturbation ${method} with severity ${severity}"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        # Launch ORB_SLAM front-end
        {
            if [ $denoise -eq 1 ]; then
                echo "Perturbation with denoise"
                python ./ablation_perturbation/src/rosbag_perturbation.py --agent_number $Perturb_agent --severity $severity --noise_type $method --trail $i --experiment $Experiment --overwrite_bag_only --noise_freq ${frequency} --noise_duration ${duration} --denoise & PERTURB_PID=$!
            else
                python ./ablation_perturbation/src/rosbag_perturbation.py --agent_number $Perturb_agent --severity $severity --noise_type $method --trail $i --experiment $Experiment --overwrite_bag_only --noise_freq ${frequency} --noise_duration ${duration} & PERTURB_PID=$!
            fi
            

            wait $PERTURB_PID
        }

        for pid in $(ps -ef | grep -e "ros" -e "rviz" | awk '{print $2}'); do kill -9 $pid; done

        echo "All processes killed and cleaned up!" & sleep 10
    done


done