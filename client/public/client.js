import { io } from "./socket.io.esm.min.js";

// Replace with your actual signaling server URL.
const signalingServerUrl = "https://western-fantasy-soybean.glitch.me";
const socket = io(signalingServerUrl);

const startBtn = document.getElementById('startBtn');
const video = document.getElementById('inputVideo');
const depthImg = document.getElementById('depthOutput');
const canvas = document.createElement('canvas');

const signalStatusEl = document.getElementById('signalStatus');
const inferenceStatusEl = document.getElementById('inferenceStatus');

let lastDepthFrameTimestamp = 0;

// Update signaling status based on connection events.
socket.on('connect', () => {
  console.log("Connected to signaling server");
  signalStatusEl.textContent = "Signaling: 🟢";
  socket.emit("register_client");
});

socket.on('disconnect', () => {
  console.log("Disconnected from signaling server");
  signalStatusEl.textContent = "Signaling: 🔴";
});

// When a depth frame is received, update inference status.
socket.on("depth_frame", (data) => {
  if (data.depth_image) {
    depthImg.src = "data:image/jpeg;base64," + data.depth_image;
    lastDepthFrameTimestamp = Date.now();
    inferenceStatusEl.textContent = "Inference: 🟢";
  }
});

// Check every second to update the inference connection status.
setInterval(() => {
  const now = Date.now();
  const elapsed = now - lastDepthFrameTimestamp;
  if (elapsed < 2000) {
    inferenceStatusEl.textContent = "Inference: 🟢";
  } else if (elapsed < 5000) {
    inferenceStatusEl.textContent = "Inference: 🟡";
  } else {
    inferenceStatusEl.textContent = "Inference: 🔴";
  }
}, 1000);

startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    startBtn.style.display = 'none';
    processFrame();
  } catch (err) {
    console.error("Error accessing webcam: ", err);
  }
});

async function processFrame() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (width && height) {
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, width, height);
    const dataURL = canvas.toDataURL('image/jpeg');
    // Emit the frame with event name "video_frame"
    socket.emit("video_frame", { image: dataURL });
  }
  // Capture a frame every 100ms; adjust the delay as needed.
  setTimeout(processFrame, 100);
}
