from flask import Flask, render_template, jsonify, request, url_for, send_file
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Define the path to store the uploaded audio files on the Desktop
user_profile = os.path.expanduser('~')
desktop_path = Path(user_profile) / 'Desktop'
audio_file_path = desktop_path / 'uploaded_audio.mp3'

# Ensure the folder exists
audio_file_path.parent.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return render_template('transcription-render-0921.html')

@app.route('/generate-transcription', methods=['POST'])
def generate_transcription():
    try:
        # Retrieve uploaded file
        audio = request.files['audio']
        if not audio:
            return jsonify({'error': 'No audio file provided'}), 400

        # Save the audio file
        audio.save(audio_file_path)

        # Call OpenAI's transcription API
        with open(audio_file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model='whisper-1',
                file=audio_file
            )

        transcript = response['text']
        return jsonify({"transcript": transcript}), 200

    except OpenAIError as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
