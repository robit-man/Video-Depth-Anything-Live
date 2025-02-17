#!/usr/bin/env python
import os
import sys
import subprocess

def in_virtualenv():
    """Return True if we’re running inside a virtual environment."""
    return sys.prefix != sys.base_prefix

if __name__ == "__main__" and not in_virtualenv():
    # Determine the absolute path for the venv directory (named "venv")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(base_dir, "venv")
    
    # Create the virtual environment if it doesn't exist
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    else:
        print("Virtual environment already exists.")
    
    # Depending on the OS, set paths for the pip and python executables
    if os.name == "nt":
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_executable = os.path.join(venv_dir, "bin", "pip")
        python_executable = os.path.join(venv_dir, "bin", "python")
    
    # Install required packages
    print("Installing requirements...")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
    
    # Re-run the script using the virtual environment's python
    print("Re-running script inside the virtual environment...")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()


# =======================================================================
# Now inside the virtual environment: import all necessary modules.
# Custom modules (video_depth_anything and utils.dc_utils) can now be imported.
# =======================================================================

import argparse
import numpy as np
import torch

def main():
    # Import custom modules now that we're in the venv.
    from video_depth_anything.video_depth import VideoDepthAnything

    parser = argparse.ArgumentParser(
        description='Video Depth Anything - Live Webcam Mode (Flask Web Frontend)'
    )
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory (unused in live mode)')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input size for depth inference')
    parser.add_argument('--max_res', type=int, default=1280,
                        help='Maximum resolution for processing (unused in live mode)')
    # Use the small model by default
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'],
                        help='Model encoder type; use "vits" for the small model')
    parser.add_argument('--fp32', action='store_true',
                        help='Run model inference in FP32 (default is FP16)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Output grayscale depth map')
    parser.add_argument('--save_npz', action='store_true',
                        help='(Optional) Save depth frames as NPZ files')
    parser.add_argument('--save_exr', action='store_true',
                        help='(Optional) Save depth frames as EXR files')
    args = parser.parse_args()

    # Determine the computation device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define model configurations with the small model as default
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Initialize and load the model using the selected encoder (small model by default)
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    checkpoint_path = f'./checkpoints/video_depth_anything_{args.encoder}.pth'
    video_depth_anything.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu'), strict=True
    )
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # ----- LIVE WEBCAM + FLASK MODE -----
    from flask import Flask, Response, render_template_string
    import cv2

    app = Flask(__name__)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Convert captured frame from BGR (OpenCV) to RGB (model expected)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run inference on the single frame (wrapped in a list)
            depths, _ = video_depth_anything.infer_video_depth(
                [frame_rgb], 30,
                input_size=args.input_size,
                device=DEVICE,
                fp32=args.fp32
            )
            depth_frame = depths[0]

            # Normalize depth frame if needed
            if depth_frame.dtype != np.uint8:
                depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_frame = depth_frame.astype(np.uint8)

            # Ensure the depth frame is 3-channel (BGR) for JPEG encoding
            if len(depth_frame.shape) == 2 or (len(depth_frame.shape) == 3 and depth_frame.shape[2] == 1):
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)

            ret, buffer = cv2.imencode('.jpg', depth_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.route('/')
    def index():
        # Simple HTML page to display the live video stream.
        return render_template_string("""
            <html>
            <head>
                <title>Live Video Depth</title>
            </head>
            <body style="background:black;display:flex;flex-flow:column;justify-content:center;margin:unset;">
                <img style="margin:auto" src="{{ url_for('video_feed') }}" width="640" height="480">
            </body>
            </html>
        """)

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        print("Starting Flask server at http://0.0.0.0:5000 ...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        camera.release()

if __name__ == '__main__':
    main()
