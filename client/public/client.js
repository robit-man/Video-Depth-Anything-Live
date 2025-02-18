import { io } from "socket.io-client";

const signalingServerUrl = "https://western-fantasy-soybean.glitch.me";
const socket = io(signalingServerUrl);

const startBtn = document.getElementById('startBtn');
const video = document.getElementById('inputVideo');
const depthImg = document.getElementById('depthOutput');

const signalStatusEl = document.getElementById('signalStatus');
const webrtcStatusEl = document.getElementById('webrtcStatus');
const fallbackStatusEl = document.getElementById('fallbackStatus');

// Mode selection buttons
const webrtcModeBtn = document.getElementById('webrtcModeBtn');
const fallbackModeBtn = document.getElementById('fallbackModeBtn');

let communicationMode = null; // "webrtc" or "fallback"
let pc; // RTCPeerConnection
let dataChannel;
let webrtcConnected = false;

socket.on('connect', () => {
  console.log("Connected to signaling server");
  signalStatusEl.textContent = "Signaling: 🟢";
  socket.emit("register_client");
});

socket.on('disconnect', () => {
  console.log("Disconnected from signaling server");
  signalStatusEl.textContent = "Signaling: 🔴";
});

// WebRTC signaling handlers.
socket.on("webrtc_answer", async (data) => {
  console.log("Received WebRTC answer");
  try {
    await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: data.sdp }));
  } catch (err) {
    console.error("Error setting remote description:", err);
  }
});

socket.on("webrtc_candidate", async (data) => {
  console.log("Received ICE candidate");
  try {
    await pc.addIceCandidate(data.candidate);
  } catch (e) {
    console.error("Error adding ICE candidate:", e);
  }
});

// Fallback: Handle depth frames from the server.
socket.on("depth_frame", (data) => {
  if (communicationMode === "fallback" && data.depth_image) {
    depthImg.src = "data:image/jpeg;base64," + data.depth_image;
    fallbackStatusEl.textContent = "Fallback (WS): 🟢";
  }
});

function createPeerConnection() {
  const config = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
  pc = new RTCPeerConnection(config);
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("webrtc_candidate", { target: "inference", candidate: event.candidate });
    }
  };
  pc.ondatachannel = (event) => {
    dataChannel = event.channel;
    setupDataChannel();
  };
}

function setupDataChannel() {
  dataChannel.onopen = () => {
    console.log("Data channel open");
    webrtcStatusEl.textContent = "WebRTC: 🟢";
    webrtcConnected = true;
  };
  dataChannel.onclose = () => {
    console.log("Data channel closed");
    webrtcStatusEl.textContent = "WebRTC: 🔴";
    webrtcConnected = false;
  };
  dataChannel.onmessage = (event) => {
    depthImg.src = "data:image/jpeg;base64," + event.data;
  };
}

async function startWebRTC() {
  createPeerConnection();
  dataChannel = pc.createDataChannel("video");
  setupDataChannel();
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  socket.emit("webrtc_offer", { sdp: offer.sdp });
}

let sendFrameInterval;
function startSendingFrames() {
  sendFrameInterval = setInterval(() => {
    if (video.readyState < 2) return;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width && height) {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);
      const dataURL = canvas.toDataURL('image/jpeg', 0.3);
      const base64Data = dataURL.split(",")[1];
      if (communicationMode === "webrtc" && webrtcConnected && dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(base64Data);
      } else if (communicationMode === "fallback") {
        // Send the full dataURL for fallback.
        socket.emit("video_frame", { image: dataURL });
      }
    }
  }, 100);
}

webrtcModeBtn.addEventListener('click', () => {
  communicationMode = "webrtc";
  webrtcStatusEl.textContent = "WebRTC: Connecting...";
  fallbackStatusEl.textContent = "Fallback (WS): (Not used)";
  console.log("Communication mode set to WebRTC");
});

fallbackModeBtn.addEventListener('click', () => {
  communicationMode = "fallback";
  fallbackStatusEl.textContent = "Fallback (WS): Waiting...";
  webrtcStatusEl.textContent = "WebRTC: (Not used)";
  console.log("Communication mode set to Fallback (WS)");
});

startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    startBtn.style.display = 'none';

    if (!communicationMode) {
      // Default to fallback if no mode selected.
      communicationMode = "fallback";
      fallbackStatusEl.textContent = "Fallback (WS): Waiting...";
    }

    if (communicationMode === "webrtc") {
      await startWebRTC();
      // Allow a few seconds for the data channel to connect.
      setTimeout(() => {
        if (!webrtcConnected) {
          console.warn("WebRTC connection failed; switching to fallback.");
          communicationMode = "fallback";
          webrtcStatusEl.textContent = "WebRTC: (Fallback)";
        }
        startSendingFrames();
      }, 3000);
    } else {
      startSendingFrames();
    }
  } catch (err) {
    console.error("Error accessing webcam:", err);
  }
});
