from flask import Flask, render_template, request, send_file
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get uploaded files
    source_image = request.files['source_image']
    driving_video = request.files['driving_video']
    
    # Save files locally
    source_path = 'static/source.jpg'
    driving_path = 'static/driving.mp4'
    source_image.save(source_path)
    driving_video.save(driving_path)
    
    # Generate video
    subprocess.run(['python', 'model/generate_video.py', source_path, driving_path])
    
    return send_file('static/generated_video.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
