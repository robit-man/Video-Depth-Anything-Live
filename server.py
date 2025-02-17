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
    
    # Install required packages (ensure requirements.txt includes needed packages)
    print("Installing requirements...")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
    
    # Re-run the script using the virtual environment's python
    print("Re-running script inside the virtual environment...")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()


import argparse
import numpy as np
import torch
import base64
import cv2
import socketio  # Requires python-socketio[client]

def main():
    from video_depth_anything.video_depth import VideoDepthAnything
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose

    parser = argparse.ArgumentParser(
        description='Inference Server with Socket.IO Signaling (Fast Mode)'
    )
    parser.add_argument('--input_size', type=int, default=518,
                        help='Base input size for depth inference')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'],
                        help='Model encoder type; use "vits" for lower memory usage')
    parser.add_argument('--fp32', action='store_true',
                        help='Run model inference in FP32 (default is FP16)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Output grayscale depth map')
    parser.add_argument('--fast_scale', type=float, default=1,
                        help='Scale factor to reduce resolution for fast inference')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    checkpoint_path = f'./checkpoints/video_depth_anything_{args.encoder}.pth'
    video_depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    fast_input_size = int(args.input_size * args.fast_scale)
    fast_transform = Compose([
        Resize(
            width=fast_input_size,
            height=fast_input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    def fast_infer(frame):
        h, w = frame.shape[:2]
        transformed = fast_transform({'image': frame.astype(np.float32) / 255.0})['image']
        input_tensor = torch.from_numpy(transformed).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, enabled=(not args.fp32)):
                depth = video_depth_anything.forward(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )
        return depth[0, 0].cpu().numpy()

    sio = socketio.Client()

    @sio.event
    def connect():
        print("Connected to signaling server")
        sio.emit("register_inference")

    @sio.event
    def disconnect():
        print("Disconnected from signaling server")

    @sio.on("frame")
    def on_frame(data):
        # Expect data to contain a key "image" and optionally "client_id".
        if "image" not in data:
            return
        client_id = data.get("client_id", None)
        img_data = data["image"]
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]
        try:
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_frame = fast_infer(frame_rgb)
            if depth_frame.dtype != np.uint8:
                depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_frame = depth_frame.astype(np.uint8)
            if not args.grayscale:
                if len(depth_frame.shape) == 2 or (len(depth_frame.shape) == 3 and depth_frame.shape[2] == 1):
                    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
            ret, buffer = cv2.imencode('.jpg', depth_frame)
            if not ret:
                return
            result_b64 = base64.b64encode(buffer).decode('utf-8')
            # Emit depth_frame with the same client_id so that the signaling server can forward it correctly.
            sio.emit("depth_frame", {"depth_image": result_b64, "client_id": client_id})
        except Exception as e:
            print("Error processing frame:", e)

    # Also listen for "video_frame" events and handle them the same way.
    @sio.on("video_frame")
    def on_video_frame(data):
        on_frame(data)

    glitch_url = "https://western-fantasy-soybean.glitch.me"
    try:
        sio.connect(glitch_url)
    except Exception as e:
        print("Failed to connect to signaling server. Ensure the signaling server is running and the URL is correct.", e)
        sys.exit(1)
    print("Inference server running. Connected to signaling server.")
    sio.wait()

if __name__ == '__main__':
    main()
