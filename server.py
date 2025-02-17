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


# =======================================================================
# Inference Server with WebRTC Data Channel and Socket.IO Signaling
# =======================================================================

#!/usr/bin/env python
import os
import sys
import subprocess
import asyncio
import base64
import numpy as np
import cv2
import torch
import socketio  # Uses the default asyncio client from python-socketio
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

# --- VENV AUTO-STARTER ---
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
    
    # Install required packages (ensure requirements.txt includes needed packages, including aiohttp)
    print("Installing requirements...")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
    
    # Re-run the script using the virtual environment's python
    print("Re-running script inside the virtual environment...")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()
# --- End VENV AUTO-STARTER ---

# =======================================================================
# Inference Server with WebRTC Data Channel and Socket.IO Signaling
# =======================================================================

import argparse
from torchvision.transforms import Compose
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Global variables to be set in main.
model = None
DEVICE = None
fast_transform = None
ARGS = None  # Global container for command-line arguments

def load_model(args):
    device_local = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    mdl = VideoDepthAnything(**model_configs[args.encoder])
    checkpoint_path = f'./checkpoints/video_depth_anything_{args.encoder}.pth'
    mdl.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    mdl = mdl.to(device_local).eval()
    return mdl, device_local

def create_fast_transform(args):
    fast_input_size = int(args.input_size * args.fast_scale)
    return Compose([
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

def fast_infer(frame, model, DEVICE, fast_transform, args):
    h, w = frame.shape[:2]
    transformed = fast_transform({'image': frame.astype(np.float32) / 255.0})['image']
    input_tensor = torch.from_numpy(transformed).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, enabled=(not args.fp32)):
            depth = model.forward(input_tensor)
    depth = torch.nn.functional.interpolate(
        depth.flatten(0, 1).unsqueeze(1),
        size=(h, w),
        mode='bilinear',
        align_corners=True
    )
    return depth[0, 0].cpu().numpy()

# Create an asyncio-based Socket.IO client.
sio = socketio.AsyncClient()

# Dictionary mapping client_id to RTCPeerConnection.
peer_connections = {}

@sio.event
async def connect():
    print("Connected to signaling server")
    await sio.emit("register_inference")

@sio.event
async def disconnect():
    print("Disconnected from signaling server")

@sio.on("webrtc_offer")
async def on_webrtc_offer(data):
    client_id = data["client_id"]
    sdp = data["sdp"]
    print(f"Received WebRTC offer from client {client_id}")
    pc = RTCPeerConnection()
    peer_connections[client_id] = pc

    # When a data channel is established from the client.
    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"Data channel established for client {client_id}")
        @channel.on("message")
        async def on_message(message):
            try:
                # Message is a base64-encoded JPEG frame.
                img_bytes = base64.b64decode(message)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Use the global ARGS variable.
                depth_frame = fast_infer(frame_rgb, model, DEVICE, fast_transform, ARGS)
                if depth_frame.dtype != np.uint8:
                    depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                    depth_frame = depth_frame.astype(np.uint8)
                if not ARGS.grayscale:
                    if len(depth_frame.shape) == 2 or (len(depth_frame.shape) == 3 and depth_frame.shape[2] == 1):
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                ret, buffer = cv2.imencode('.jpg', depth_frame)
                if not ret:
                    return
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                channel.send(result_b64)
            except Exception as e:
                print(f"Error processing data channel message for client {client_id}:", e)

    # Set remote description using the received offer.
    await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await sio.emit("webrtc_answer", {
        "client_id": client_id,
        "sdp": pc.localDescription.sdp
    })

    @pc.on("icecandidate")
    async def on_icecandidate(event):
        if event.candidate:
            # Use positional arguments when creating ICE candidate.
            candidate = RTCIceCandidate(
                event.candidate.to_sdp(),
                event.candidate.sdpMid,
                event.candidate.sdpMLineIndex
            )
            candidate_dict = {
                "candidate": str(candidate),
                "sdpMid": event.candidate.sdpMid,
                "sdpMLineIndex": event.candidate.sdpMLineIndex,
            }
            await sio.emit("webrtc_candidate", {
                "target": "client",
                "client_id": client_id,
                "candidate": candidate_dict
            })

@sio.on("webrtc_candidate")
async def on_webrtc_candidate(data):
    client_id = data.get("client_id")
    candidate = data.get("candidate")
    if client_id in peer_connections and candidate:
        pc = peer_connections[client_id]
        try:
            ice_candidate = RTCIceCandidate(
                candidate["candidate"],
                candidate["sdpMid"],
                candidate["sdpMLineIndex"]
            )
            await pc.addIceCandidate(ice_candidate)
        except Exception as e:
            print(f"Error adding ICE candidate for client {client_id}:", e)

# Fallback: Handle frames received over Socket.IO.
@sio.on("frame")
async def on_frame(data):
    client_id = data.get("client_id")
    if "image" not in data:
        return
    img_data = data["image"]
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]
    try:
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_frame = fast_infer(frame_rgb, model, DEVICE, fast_transform, ARGS)
        if depth_frame.dtype != np.uint8:
            depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_frame = depth_frame.astype(np.uint8)
        if not ARGS.grayscale:
            if len(depth_frame.shape) == 2 or (len(depth_frame.shape) == 3 and depth_frame.shape[2] == 1):
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        ret, buffer = cv2.imencode('.jpg', depth_frame)
        if not ret:
            return
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        await sio.emit("depth_frame", {"depth_image": result_b64, "client_id": client_id})
    except Exception as e:
        print("Error processing fallback frame:", e)

async def main():
    parser = argparse.ArgumentParser(
        description="Inference Server with WebRTC Data Channel"
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

    global model, DEVICE, fast_transform, ARGS
    ARGS = args  # Save arguments globally.
    model, DEVICE = load_model(args)
    fast_transform = create_fast_transform(args)

    # Connect to the signaling server.
    await sio.connect("https://western-fantasy-soybean.glitch.me")
    print("Inference server connected to signaling server.")
    await sio.emit("register_inference")
    await sio.wait()

if __name__ == '__main__':
    asyncio.run(main())
