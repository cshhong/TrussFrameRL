import pandas as pd
import os
'''
multiple_path results process

'''
def process_csv(results_csv):
    # Step 1: Read the CSV file
    # input_csv = os.path.join(results_dir, "results.csv")
    input_csv = results_csv
    results_dir = os.path.dirname(input_csv)
    print(f"Processing CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    # print(f'{df}')

    # Step 2: Group by "Boundary Condition"
    grouped = df.groupby("Boundary Condition")

    # Step 3: Process each group and create summary
    summary_data = []


    for boundary_condition, group in grouped:
        # take the first 200 instances
        group = group.iloc[:200]
        # print(f"Processing boundary condition: {boundary_condition}")
        # print(f'{group}')
        # Add the number of data points for this group
        num_datapoints = len(group)
        num_terminated = (group["Terminated"] == True).sum() 
        num_truncated = (group["Terminated"] == False).sum()

        terminated_group = group[group["Terminated"] == True]# Filter rows where "Terminated" is True
        
        # Calculate the number of unique solutions
        # filter rows within terminated_group that are unique
        unique_solutions_group = terminated_group.drop_duplicates(subset=["Frame Grid"])
        num_unique_solutions = len(unique_solutions_group)

        # Unique Solutions Failed Elements
        unique_failed_elem_mean = unique_solutions_group["Number of Failed Elements"].mean() 
        unique_failed_elem_std = unique_solutions_group["Number of Failed Elements"].std()

        # Unique Solutions Frame Count
        unique_frame_count_mean = unique_solutions_group["Number of Frames"].mean()
        unique_frame_count_std = unique_solutions_group["Number of Frames"].std()

        # filter rows within unique solutions group that have failed elements
        # unique_failed_group = unique_solutions_group[unique_solutions_group["Number of Failed Elements"] > 0]# Filter rows where "Number of Failed Elements" is greater than 0
        # num_eps_with_failed = unique_failed_group.shape[0]
        # avg_failed = unique_failed_group["Number of Failed Elements"].mean() if num_eps_with_failed > 0 else 0
        # std_failed = unique_failed_group["Number of Failed Elements"].std() if num_eps_with_failed > 0 else 0

        # Filter designs without failed elements
        unique_nofail_group = unique_solutions_group[unique_solutions_group["Number of Failed Elements"] == 0]
        # Calculate the number of unique solutions without failed elements
        num_nofail_solutions = len(unique_nofail_group)

        # print(f' nofail solution group : {nofail_solution_group}')
        # Calculate averages for the required columns
        averages = unique_nofail_group[["Max Deflection", "Utilization Median", "Utilization Std", "Number of Frames", "Utilization P90"]].mean()
        perc_util_med = averages["Utilization Median"]*100 
        perc_util_std = averages["Utilization Std"]*100 
        perc_util_p90 = averages["Utilization P90"]*100
        perc_util_p90_std = unique_nofail_group["Utilization P90"].std()*100

        # Unique Solutions within allowable deflection
        unique_within_defl_group = unique_solutions_group[unique_solutions_group["Max Deflection"] <= group["Allowable Deflection"].iloc[0]]
        num_within_defl_solutions = len(unique_within_defl_group)
        unique_within_defl_max_deflection_mean = unique_within_defl_group["Max Deflection"].mean()
        unique_within_defl_max_deflection_std = unique_within_defl_group["Max Deflection"].std()

        # Append the summary data for this boundary condition
        summary_data.append({
            "Boundary Condition": boundary_condition,
            "Allowable Deflection": round(group["Allowable Deflection"].iloc[0], 3),
            "Number of Data Points": num_datapoints,
            "Number of Truncated" : num_truncated,
            "Number of Terminated" : num_terminated,
            "Number of Unique Solutions": num_unique_solutions,
            "Unique Solutions Avg Failed Elements": round(unique_failed_elem_mean,2),
            "Unique Solutions Std Failed Elements": round(unique_failed_elem_std,2),
            "Unique Solutions Avg Frame Count": round(unique_frame_count_mean,2),
            "Unique Solutions Std Frame Count": round(unique_frame_count_std,2),
            "Number of Unique Solutions w/o Failed Elements": num_nofail_solutions,
            "Unique w/o Failed Avg Utilization 90 Percentile": round(perc_util_p90, 3),
            "Unique w/o Failed Std Utilization 90 Percentile": round(perc_util_p90_std, 3),
            "Number of Unique Solutions within Allowable Deflection": num_within_defl_solutions,
            "Unique Solutions Avg Max Deflection": round(unique_within_defl_max_deflection_mean, 3),
            "Unique Solutions Std Max Deflection": round(unique_within_defl_max_deflection_std, 3),

            # "Number of Episodes w/ Failed Elements": num_eps_with_failed,
            # "Number of Episodes w/o Failed Elements": num_terminated-num_eps_with_failed,
            # "Average Number of Failed Elements": round(avg_failed, 2),
            # "Number of Failed Elements Std": round(std_failed, 2),
            # "Number of Unique Solutions without Failure" : num_nofail_solutions,
            # "Allowable Deflection": round(group["Allowable Deflection"].iloc[0], 3),
            # "Average Max Deflection": round(averages["Max Deflection"], 3),
            # "Max Deflection Std": round(group["Max Deflection"].std(), 3),
            # "Average Utilization Median": round(perc_util_med, 3),
            # "Average Utilization Std": round(perc_util_std, 3),
            # "Average Utilization 90 Percentile": round(perc_util_p90, 3),
            # "Average Number of Frames": round(averages["Number of Frames"],3),
            # "Std Number of Frames": round(nofail_solution_group["Number of Frames"].std(),2),
        })

        # create a separate csv file that has each unique solution
        unique_solutions_group.to_csv(os.path.join(results_dir, f"unique_solutions.csv"), index=False)
        print(f"Saving unique solutions for boundary condition: {boundary_condition} in {os.path.join(results_dir, f'unique_solutions.csv')}")

    # Create a DataFrame from the summ dary data
    summary_df = pd.DataFrame(summary_data)

    # Write the summary DataFrame to a single CSV file
    org_name = os.path.basename(input_csv)
    org_name = org_name.split(".")[0] # remove end .csv
    output_csv = os.path.join(results_dir, f"{org_name}_summary.csv")  # Replace with your desired output file name
    summary_df.to_csv(output_csv, index=False)

    print(f"Summary results saved in '{output_csv}'.")

if __name__ == "__main__":
    # Singular usage
    results_dir = "May10_allbccompare/32_h4l5m150-h3l2m0/inference/results.csv"
    process_csv(results_dir)
    
    # all_runs_dir = "Apr24_results"
    # # for all subdirectories in the directory
    # dirs = [os.path.join(all_runs_dir, d) for d in os.listdir(all_runs_dir) if os.path.isdir(os.path.join(all_runs_dir, d))]
    # for dir in dirs:
    #     inference_dir = os.path.join(dir, "inference")
    #     # Check if the directory exists
    #     if os.path.exists(inference_dir):
    #         print(f"Processing directory: {inference_dir}")
    #         process_csv(inference_dir)
    #     else:
    #         print(f"Directory does not exist: {inference_dir}")