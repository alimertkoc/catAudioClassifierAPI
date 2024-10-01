import requests

def test_predict_api(api_url: str, audio_file_path: str):
    """
    Sends an audio file to the Cat Meow Classifier API and prints the response.

    :param api_url: The URL of the prediction endpoint.
    :param audio_file_path: The path to the audio file to be sent.
    """
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (audio_file_path, audio_file, 'audio/wav')}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            print("Prediction Response:", response.json())
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
    
    except FileNotFoundError:
        print(f"Audio file not found at path: {audio_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    API_URL = "http://0.0.0.0:8000/predict"  # Update if your API is hosted elsewhere
    B_AUDIO_FILE_PATH = "/Users/alimert/Downloads/projects/catBackend/B_BRA01_MC_MN_SIM01_301.wav"  # Replace with the path to your audio file
    F_AUDIO_FILE_PATH = "/Users/alimert/Downloads/projects/catBackend/F_BAC01_MC_MN_SIM01_202.wav"  # Replace with the path to your audio file
    I_AUDIO_FILE_PATH = "/Users/alimert/Downloads/projects/catBackend/I_WHO01_MC_FI_SIM01_201.wav"  # Replace with the path to your audio file
    # Test the API with the provided audio file
    
    print("Testing B_BRA01_MC_MN_SIM01_301.wav")
    test_predict_api(API_URL, B_AUDIO_FILE_PATH)

    print("Testing F_BAC01_MC_MN_SIM01_202.wav")
    test_predict_api(API_URL, F_AUDIO_FILE_PATH)

    print("Testing I_WHO01_MC_FI_SIM01_201.wav")
    test_predict_api(API_URL, I_AUDIO_FILE_PATH)