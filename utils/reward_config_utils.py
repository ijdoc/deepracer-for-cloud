import numpy as np
import requests
from io import BytesIO


def download(track):
    # URL of the npy file
    npy_url = f"https://github.com/aws-deepracer-community/deepracer-race-data/raw/main/raw_data/tracks/npy/{track}.npy"

    # Download the npy file
    response = requests.get(npy_url)
    if response.status_code == 200:
        # Load the npy file and count the number of inner_line
        npy_data = np.load(BytesIO(response.content))
        # Save to .npy file
        np.save(f"{track}.npy", npy_data)
    else:
        waypoint_count = "Error downloading npy file"

    print(f"Downloaded {track} with {npy_data.shape[0]} waypoints.")
    return npy_data
