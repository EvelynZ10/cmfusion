import pandas as pd
import pickle


def prepare_and_save_data(csv_file_path, output_pickle_path):
    # Load the data from CSV
    data = pd.read_csv(csv_file_path)

    # Map textual labels to integer labels
    label_mapping = {'Hateful': 0, 'Offensive': 1, 'Normal': 2}
    data['Majority_Voting'] = data['Majority_Voting'].map(label_mapping)

    # Creating separate lists for video names and labels
    video_names = data['Video_ID'].tolist()
    labels = data['Majority_Voting'].tolist()

    # Combining video names and labels into a dictionary
    data_dict = {'all_data': (video_names, labels)}

    # Save the dictionary as a pickle file
    with open(output_pickle_path, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Data has been successfully saved to {output_pickle_path}")


# Example usage
csv_file_path = 'multihateclip_english_fold.csv'  # Replace with your actual CSV file path
output_pickle_path = 'multihateclip_final_allNewData.p'  # Desired output pickle file name
prepare_and_save_data(csv_file_path, output_pickle_path)
