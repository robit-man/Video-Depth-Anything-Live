import { io } from "./socket.io.esm.min.js";

const signalingServerUrl = "https://western-fantasy-soybean.glitch.me";
const socket = io(signalingServerUrl);

const startBtn = document.getElementById('startBtn');
const video = document.getElementById('inputVideo');
const depthImg = document.getElementById('depthOutput');
const signalStatusEl = document.getElementById('signalStatus');
const webrtcStatusEl = document.getElementById('webrtcStatus');

let pc; // RTCPeerConnection
let dataChannel;

socket.on('connect', () => {
  console.log("Connected to signaling server");
  signalStatusEl.textContent = "Signaling: 🟢";
  socket.emit("register_client");
});

socket.on('disconnect', () => {
  console.log("Disconnected from signaling server");
  signalStatusEl.textContent = "Signaling: 🔴";
});

socket.on("webrtc_answer", async (data) => {
  console.log("Received WebRTC answer");
  const sdp = data.sdp;
  await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp }));
});

socket.on("webrtc_candidate", async (data) => {
  console.log("Received ICE candidate");
  try {
    await pc.addIceCandidate(data.candidate);
  } catch (e) {
    console.error("Error adding ICE candidate:", e);
  }
});

// Create the RTCPeerConnection and data channel.
function createPeerConnection() {
  const config = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
  pc = new RTCPeerConnection(config);
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("webrtc_candidate", { target: "inference", candidate: event.candidate });
    }
  };
  // Listen for the data channel created by the inference server.
  pc.ondatachannel = (event) => {
    dataChannel = event.channel;
    setupDataChannel();
  };
}

function setupDataChannel() {
  dataChannel.onopen = () => {
    console.log("Data channel open");
    webrtcStatusEl.textContent = "WebRTC: 🟢";
    startSendingFrames();
  };
  dataChannel.onclose = () => {
    console.log("Data channel closed");
    webrtcStatusEl.textContent = "WebRTC: 🔴";
  };
  dataChannel.onmessage = (event) => {
    // Assume the inference server sends back a base64 JPEG.
    depthImg.src = "data:image/jpeg;base64," + event.data;
  };
}

async function startWebRTC() {
  createPeerConnection();
  // Create a data channel for sending video frames.
  dataChannel = pc.createDataChannel("video");
  setupDataChannel();

  // Create an SDP offer and send it via signaling.
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  socket.emit("webrtc_offer", { sdp: offer.sdp });
}

let sendFrameInterval;
function startSendingFrames() {
  sendFrameInterval = setInterval(() => {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width && height) {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);
      // Use a lower quality setting (0.3) to reduce message size.
      const dataURL = canvas.toDataURL('image/jpeg', 0.3);
      // Remove the data URL prefix.
      const base64Data = dataURL.split(",")[1];
      if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(base64Data);
      }
    }
  }, 100); // Send a frame every 100ms.
}

startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    startBtn.style.display = 'none';
    await startWebRTC();
  } catch (err) {
    console.error("Error accessing webcam:", err);
  }
});
