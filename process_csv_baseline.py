import pandas as pd
import os
'''
multiple_path results process

'''
def process_csv(results_dir=None, csv_path=None):
    # Step 1: Read the CSV file
    if results_dir != None and csv_path == None:
        input_csv = os.path.join(results_dir, "results.csv")
    
    if csv_path != None:
        if results_dir == None: # make sure there is alsp results_dir explicitly given
            raise ValueError("results_dir must be provided if csv_path is given")
        else:
            input_csv = csv_path

    print(f"Processing CSV file: {input_csv}")
    df = pd.read_csv(input_csv)

    # Step 2: Group by "Boundary Condition"
    grouped = df.groupby("Boundary Condition")

    # Step 3: Process each group and create summary
    summary_data = []


    for boundary_condition, group in grouped:
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

        # filter rows within unique solutions group that have failed elements
        failed_group = unique_solutions_group[unique_solutions_group["Number of Failed Elements"] > 0]# Filter rows where "Number of Failed Elements" is greater than 0
        num_eps_with_failed = failed_group.shape[0]
        avg_failed = failed_group["Number of Failed Elements"].mean() if num_eps_with_failed > 0 else 0
        std_failed = failed_group["Number of Failed Elements"].std() if num_eps_with_failed > 0 else 0

        # Filter designs without failed elements
        nofail_solution_group = unique_solutions_group[unique_solutions_group["Number of Failed Elements"] == 0]
        # Calculate the number of unique solutions without failed elements
        num_nofail_solutions = len(nofail_solution_group)
        
        # print(f' nofail solution group : {nofail_solution_group}')
        # Calculate averages for the required columns
        averages = nofail_solution_group[["Max Deflection", "Utilization Median", "Utilization Std", "Utilization P90", "Number of Frames"]].mean()
        # Append the summary data for this boundary condition
        summary_data.append({
            "Boundary Condition": boundary_condition,
            "Number of Data Points": num_datapoints,
            "Number of Truncated" : num_truncated,
            "Number of Terminated" : num_terminated,
            "Number of Episodes w/ Failed Elements": num_eps_with_failed,
            "Number of Episodes w/o Failed Elements": num_terminated-num_eps_with_failed,
            "Average Number of Failed Elements": round(avg_failed, 2),
            "Number of Failed Elements Std": round(std_failed, 2),
            "Number of Unique Solutions without Failure" : num_nofail_solutions,
            "Allowable Deflection": round(group["Allowable Deflection"].iloc[0], 3),
            "Average Max Deflection": round(averages["Max Deflection"], 3),
            "Max Deflection Std": round(nofail_solution_group["Max Deflection"].std(), 3),
            "Average Utilization Median": round(averages["Utilization Median"], 3),
            "Average Utilization Std": round(averages["Utilization Std"], 3),
            "Utilization 90 Percentile" : round(averages["Utilization P90"], 3),
            "Average Number of Frames": round(averages["Number of Frames"],3),
            "Std Number of Frames": round(nofail_solution_group["Number of Frames"].std(),0),
        })

    # Create a DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)

    # Write the summary DataFrame to a single CSV file
    output_csv = os.path.join(results_dir, "baseline_summary.csv")  # Replace with your desired output file name
    summary_df.to_csv(output_csv, index=False)

    print(f"Summary results saved in '{output_csv}'.")

if __name__ == "__main__":
    # Example usage
    # results_dir = "baseline/hong_ppo_cantilever" 
    # results_dir = "Apr24_results/transfer_framecountpenalty_bc-fixed(h2l5m150)-(h3l2m0)_inv30-0/inference" 
    csv_path = "May1_results/h2l5m200-h5l0m0/baseline/baseline.csv"
    results_dir = "May1_results/h2l5m200-h5l0m0/baseline"
    process_csv(results_dir=results_dir, csv_path=csv_path)
    
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