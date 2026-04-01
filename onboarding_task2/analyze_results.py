import csv
import os
from collections import Counter

def main():
    results_path = "./trajectories/automated_demonstrations/results.csv"
    
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return

    total_episodes = 0
    success_count = 0
    failure_types = []

    print(f"Analyzing {results_path}...\n")

    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_episodes += 1
            
            # Check success (handle string 'True'/'False')
            is_success = row['success'].lower() == 'true'
            
            if is_success:
                success_count += 1
            else:
                f_type = row.get('failure_type', '').strip()
                if not f_type:
                    f_type = "unspecified"
                failure_types.append(f_type)

    if total_episodes == 0:
        print("No data found.")
        return

    success_rate = (success_count / total_episodes) * 100
    
    print("-" * 30)
    print(f"Total Episodes:   {total_episodes}")
    print(f"Successful:       {success_count}")
    print(f"Success Rate:     {success_rate:.2f}%")
    print("-" * 30)
    
    if failure_types:
        print("\nFailure Types:")
        counts = Counter(failure_types)
        for f_type, count in counts.most_common():
            print(f"  - {f_type}: {count}")
    else:
        print("\nNo failures recorded.")
    print("-" * 30)

if __name__ == "__main__":
    main()
