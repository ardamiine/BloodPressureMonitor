import os
import json

# specify the directory
path_to_json = 'path_to_folder_that_contains_both_json_and_mp4_files'
#path_to_folder_that_contains_both_json_and_mp4_files

ppg_list = []
bp_list = []

# walk through the directory
for dirpath, dirnames, filenames in os.walk(path_to_json):

    for filename in filenames:

        if filename.endswith('.json'):

            with open(os.path.join(dirpath, filename)) as json_file:

                data = json.load(json_file)
                for scenario in data['scenarios']:

                    scenario_settings = scenario['scenario_settings']

                    # Check if the scenario settings match the required criteria (this will exclude ~ 10 videos that contain talking / on purpose head movement)
                    if (scenario_settings['position'] == "Sitting" and
                            scenario_settings['facial_movement'] == "No movement" and
                            scenario_settings['talking'] == "N"):

                        # Check if the recording file exists in the directory (some videos were unreadable so they had to be deleted)
                        if scenario['recordings']['RGB']['filename'] in filenames:

                            # Construct the object for the ppg recording
                            ppg_recording = {
                                "participant_metadata": {
                                    **data['participant'],
                                    'GUID': data['GUID']
                                },
                                # timeseries contains a list of [timestamp, value] pairs. We only want the values so we extract the 2nd item from the inner list.
                                "ppg_values": [item[1] for item in scenario['recordings']['ppg']['timeseries']],
                                "recording_link": scenario['recordings']['RGB']['filename']
                            }
                            ppg_list.append(ppg_recording)

                            # Check if bp_sys and bp_dia exist in the recordings
                            if 'bp_sys' in scenario['recordings'] and 'bp_dia' in scenario['recordings']:
                                # Construct the object for the blood pressure recording
                                bp_recording = {
                                    "participant_metadata": {
                                        **data['participant'],
                                        'GUID': data['GUID']
                                    },
                                    "ppg_values": [item[1] for item in scenario['recordings']['ppg']['timeseries']],
                                    "bp_values": {
                                        "bp_sys": scenario['recordings']['bp_sys']['value'],
                                        "bp_dia": scenario['recordings']['bp_dia']['value'],
                                    },
                                    "recording_link": scenario['recordings']['RGB']['filename']
                                }
                                bp_list.append(bp_recording)

print("PPG list: ", ppg_list)
print("BP list: ", bp_list)

print("PPG list: ", len(ppg_list)) # results in 1777 for the complete dataset (should be around 200 for vv-small and around 500 for vv-medium)
print("BP list: ", len(bp_list)) # results in 866 for the complete dataset (should be around 100 for vv-small and around 250 for vv-medium)