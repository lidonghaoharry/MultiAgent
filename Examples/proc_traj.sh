#!/bin/bash
dataset="EuRoC"
BAG_PATH="/home/lidonghao/MultiAgent/data/EuRoC"
TEST_AGENTS=("MH_01_easy" "MH_02_easy")
Perturb_agent=0
# METHODS=(""gaussian_blur" "defocus_blur" "fog" "frost" 
# "contrast" "brightness" "snow" "spatter")
output_path="/home/lidonghao/ws/covins_ws/src/covins/covins_backend/output/"
Experiment="Perturb_n0_bmethods"

# overall_log_path="${output_path}${Experiment}/experiment_log.csv"
if [ ! -d ${output_path}${Experiment} ]; then
    echo "No output with name ${Experiment}, terminating"
    exit 1
fi

# BMETHODS=("gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "contrast" "pixelate" "jpeg_compression")
BMETHODS=("gaussian_noise" )

KF_0_gt=${output_path}MH_01_easy/mav0/state_groundtruth_estimate0/data.tum
KF_1_gt=${output_path}MH_02_easy/mav0/state_groundtruth_estimate0/data.tum



for method in ${BMETHODS[*]};
do
    for severity in {5..6..2};
    do  
        KF_0_output_list=()
        KF_1_output_list=()

        for i in {2,3,7,8,10};
        do 
            # Check if the output directory exists, if yes, add the output to the list
            if [ ! -d ${output_path}${Experiment}/${method}_s${severity}_trial${i} ]; then
                echo "No output with name ${Experiment}/${method}_s${severity}_trial${i}, skipping"
                continue
            fi
            log_dir="${output_path}${Experiment}/${method}_s${severity}_trial${i}/"

            KF_0_output=${log_dir}KF_0_ftum.csv
            KF_1_output=${log_dir}KF_1_ftum.csv

            KF_0_output_list+=(${KF_0_output})
            KF_1_output_list+=(${KF_1_output})

        done
        # Check if there are any completed trials
        if [ ${#KF_0_output_list[@]} -eq 0 ] || [ ${#KF_1_output_list[@]} -eq 0 ]; then
            echo "No output with name ${Experiment}/${method}_s${severity} has no completed trials, skipping"
            continue
        fi
        # If the output directory does not exist, create it
        if [ ! -d ${output_path}${Experiment}/plot_res/${method}_s${severity} ]; then
            mkdir -p ${output_path}${Experiment}/plot_res/${method}_s${severity}
        fi
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo "list: ${KF_0_output_list[*]}"
        echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        evo_traj tum ${KF_0_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_0_xz.png --plot_mode xz --ref ${KF_0_gt} 
        evo_traj tum ${KF_1_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_1_xz.png --plot_mode xz --ref ${KF_1_gt} 
        evo_traj tum ${KF_0_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_0_yz.png --plot_mode yz --ref ${KF_0_gt} 
        evo_traj tum ${KF_1_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_1_yz.png --plot_mode yz --ref ${KF_1_gt} 
        evo_traj tum ${KF_0_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_0_xy.png --plot_mode xy --ref ${KF_0_gt} 
        evo_traj tum ${KF_1_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_1_xy.png --plot_mode xy --ref ${KF_1_gt} 
        evo_traj tum ${KF_0_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_0_xyz.png --plot_mode xyz --ref ${KF_0_gt} 
        evo_traj tum ${KF_1_output_list[*]} -vas --save_plot ${output_path}${Experiment}/plot_res/${method}_s${severity}/KF_1_xyz.png --plot_mode xyz --ref ${KF_1_gt} 

    done

done
