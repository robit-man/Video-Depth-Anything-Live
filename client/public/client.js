import { io } from "https://cdn.jsdelivr.net/npm/socket.io-client@4.4.1/dist/socket.io.esm.min.js";
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.119.1/build/three.module.min.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.119.1/examples/jsm/controls/OrbitControls.min.js";
import { VRButton } from "https://cdn.jsdelivr.net/npm/three@0.119.1/examples/jsm/webxr/VRButton.min.js";
import { XRControllerModelFactory } from "https://cdn.jsdelivr.net/npm/three@0.119.1/examples/jsm/webxr/XRControllerModelFactory.min.js";

/* 
   Combined client logic with fallback (WS) vs WebRTC modes, plus Three.js XR scene.
   - Communication mode can be "webrtc" or "fallback" chosen by user.
   - If WebRTC fails, fallback to websockets automatically.
   - On each frame, we send the image to the inference server either via DataChannel or fallback socket.
   - The inference server returns a depth frame, stored in #depthOutput.
   - Meanwhile, a dynamic Three.js scene creates a geometry from #depthOutput + #inputVideo.
   - Two sliders: 
        - #pixelScaleSlider changes how far in XY each point is spaced.
        - #sizeslider changes how big each point is.
   - Another slider #depthslider changes the Z scale factor for the depth.
   - A checkbox (#singlemesh) toggles between a continuous planar mesh (with connected triangles)
     and the original points-based grid.
*/

// ----------------------------------------------------------------
// HTML element references & Socket.IO
// ----------------------------------------------------------------
const signalingServerUrl = "https://western-fantasy-soybean.glitch.me";
const socket = io(signalingServerUrl);

// UI elements
const startBtn = document.getElementById('startBtn');
const videoEl = document.getElementById('inputVideo');
const depthImgEl = document.getElementById('depthOutput');

const signalStatusEl = document.getElementById('signalStatus');
const webrtcStatusEl = document.getElementById('webrtcStatus');
const fallbackStatusEl = document.getElementById('fallbackStatus');
const webrtcModeBtn = document.getElementById('webrtcModeBtn');
const fallbackModeBtn = document.getElementById('fallbackModeBtn');

// Communication mode: "webrtc" or "fallback"
let communicationMode = null; 
let pc; // RTCPeerConnection
let dataChannel;
let webrtcConnected = false;

// Join the signaling server
socket.on('connect', () => {
  console.log("Connected to signaling server");
  signalStatusEl.textContent = "Signaling: 🟢";
  socket.emit("register_client");
});
socket.on('disconnect', () => {
  console.log("Disconnected from signaling server");
  signalStatusEl.textContent = "Signaling: 🔴";
});

// ---------------------------
// WebRTC-Specific Signaling
// ---------------------------
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
  if (!pc) return;
  try {
    await pc.addIceCandidate(data.candidate);
  } catch (e) {
    console.error("Error adding ICE candidate:", e);
  }
});

// ---------------------------
// Fallback Depth Frames
// ---------------------------
socket.on("depth_frame", (data) => {
  // Only process if in fallback mode
  if (communicationMode === "fallback" && data.depth_image) {
    depthImgEl.src = "data:image/jpeg;base64," + data.depth_image;
    fallbackStatusEl.textContent = "Fallback (WS): 🟢";
  }
});

// ---------------------------
// Create / Setup WebRTC
// ---------------------------
function createPeerConnection() {
  const config = {
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" }
      // If you need a TURN server, add it here
    ]
  };
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
    // Depth frames in "webrtc" mode come via datachannel
    depthImgEl.src = "data:image/jpeg;base64," + event.data;
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

// ---------------------------
// Frame Sending Logic
// ---------------------------
let sendFrameInterval;
function startSendingFrames() {
  sendFrameInterval = setInterval(() => {
    if (videoEl.readyState < 2) return; // not ready
    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    if (width && height) {
      // Draw the current video frame into a canvas
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, width, height);

      // Convert to base64
      const dataURL = canvas.toDataURL('image/jpeg', 0.3);
      const base64Data = dataURL.split(",")[1];

      // Send via datachannel or fallback socket
      if (communicationMode === "webrtc" && webrtcConnected && dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(base64Data);
      } else if (communicationMode === "fallback") {
        socket.emit("video_frame", { image: dataURL });
      }
    }
  }, 100); // ~10 fps
}

// ---------------------------
// Mode Buttons
// ---------------------------
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
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    startBtn.style.display = 'none';

    // Default to fallback if user hasn't chosen a mode
    if (!communicationMode) {
      communicationMode = "fallback";
      fallbackStatusEl.textContent = "Fallback (WS): Waiting...";
    }

    if (communicationMode === "webrtc") {
      await startWebRTC();
      setTimeout(() => {
        if (!webrtcConnected) {
          console.warn("WebRTC connection failed or not connected in time; switching to fallback.");
          communicationMode = "fallback";
          webrtcStatusEl.textContent = "WebRTC: (Fallback)";
          fallbackStatusEl.textContent = "Fallback (WS): Waiting...";
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

// ----------------------------------------------------------------
// THREE.js + WebXR Scene
// ----------------------------------------------------------------
let container;
let camera, scene, renderer;
let controller1, controller2;
let controllerGrip1, controllerGrip2;
let raycaster;
const tempMatrix = new THREE.Matrix4();

let controls;
let dolly;
let cameraVector = new THREE.Vector3();
const prevGamePads = new Map();
const speedFactor = [0.1, 0.1, 0.1, 0.1];

// Geometry references
let depthMesh = null;
let depthGeometry = null;
let depthPositions = null;
let depthColors = null;

let knownDepthWidth = 0;
let knownDepthHeight = 0;

// WASD controls
let keyStates = { w: false, a: false, s: false, d: false };

initThree();
animate();

function initThree() {
  container = document.createElement('div');
  document.body.appendChild(container);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(0, 1.6, 50);

  scene.add(new THREE.HemisphereLight(0x808080, 0x606060));
  const dirLight = new THREE.DirectionalLight(0xffffff);
  dirLight.position.set(0, 200, 0);
  scene.add(dirLight);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.xr.enabled = true;
  renderer.xr.setFramebufferScaleFactor(2.0);
  container.appendChild(renderer.domElement);

  document.body.appendChild(VRButton.createButton(renderer));

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.6, 0);
  controls.update();

  // Controllers
  controller1 = renderer.xr.getController(0);
  controller1.name = "left";
  controller1.addEventListener("selectstart", onSelectStart);
  controller1.addEventListener("selectend", onSelectEnd);
  scene.add(controller1);

  controller2 = renderer.xr.getController(1);
  controller2.name = "right";
  controller2.addEventListener("selectstart", onSelectStart);
  controller2.addEventListener("selectend", onSelectEnd);
  scene.add(controller2);

  const controllerModelFactory = new XRControllerModelFactory();
  controllerGrip1 = renderer.xr.getControllerGrip(0);
  controllerGrip1.add(controllerModelFactory.createControllerModel(controllerGrip1));
  scene.add(controllerGrip1);

  controllerGrip2 = renderer.xr.getControllerGrip(1);
  controllerGrip2.add(controllerModelFactory.createControllerModel(controllerGrip2));
  scene.add(controllerGrip2);

  raycaster = new THREE.Raycaster();
  const rayGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1)
  ]);
  const line = new THREE.Line(rayGeo);
  line.name = "line";
  line.scale.z = 5;
  controller1.add(line.clone());
  controller2.add(line.clone());

  dolly = new THREE.Group();
  dolly.position.set(0, 0, 0);
  scene.add(dolly);
  dolly.add(camera, controller1, controller2, controllerGrip1, controllerGrip2);

  // WASD event listeners
  window.addEventListener('keydown', (e) => {
    if (keyStates.hasOwnProperty(e.key)) {
      keyStates[e.key] = true;
    }
  });
  window.addEventListener('keyup', (e) => {
    if (keyStates.hasOwnProperty(e.key)) {
      keyStates[e.key] = false;
    }
  });

  window.addEventListener("resize", onWindowResize);

  // NEW FEATURE: Listen for singlemesh toggle changes.
  // Assumes a checkbox input with id "singlemesh" exists on the frontend.
  const singleMeshToggle = document.getElementById("singlemesh");
  if (singleMeshToggle) {
    singleMeshToggle.addEventListener("change", () => {
      // Rebuild geometry when toggled.
      if (depthImgEl.naturalWidth && depthImgEl.naturalHeight) {
        rebuildDepthGeometry(depthImgEl.naturalWidth, depthImgEl.naturalHeight);
      }
    });
  }
}

/**
 * Rebuild geometry (pixel scale + point size) from sliders.
 * When "singlemesh" is enabled, build a continuous grid by generating indices to connect adjacent vertices.
 */
function rebuildDepthGeometry(width, height) {
  // Remove existing mesh if any
  if (depthMesh) {
    scene.remove(depthMesh);
    depthMesh.geometry.dispose();
    depthMesh.material.dispose();
    depthMesh = null;
    depthGeometry = null;
  }

  depthGeometry = new THREE.BufferGeometry();
  const numPoints = width * height;
  const positions = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  // Get pixel spacing from slider
  const xySlider = document.getElementById("pixelScaleSlider");
  const xyScaleVal = parseFloat(xySlider.value) || 0.1;

  let i3 = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      positions[i3 + 0] = (x - width / 2) * xyScaleVal;
      positions[i3 + 1] = -(y - height / 2) * xyScaleVal;
      positions[i3 + 2] = 0; // initial z; will be updated based on depth
      colors[i3 + 0] = 1.0;
      colors[i3 + 1] = 1.0;
      colors[i3 + 2] = 1.0;
      i3 += 3;
    }
  }
  depthGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  depthGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // If in singlemesh mode, build indices to form a continuous grid mesh.
  const singleMeshToggle = document.getElementById("singlemesh");
  if (singleMeshToggle && singleMeshToggle.checked) {
    const indices = [];
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const i = y * width + x;
        // First triangle of the quad
        indices.push(i, i + width, i + 1);
        // Second triangle of the quad
        indices.push(i + width, i + width + 1, i + 1);
      }
    }
    depthGeometry.setIndex(indices);
  }

  depthGeometry.computeBoundingSphere();

  let material;
  if (singleMeshToggle && singleMeshToggle.checked) {
    material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide
    });
    depthMesh = new THREE.Mesh(depthGeometry, material);
  } else {
    const sizeSlider = document.getElementById("sizeslider");
    const pointSize = parseFloat(sizeSlider.value) || 0.5;
    material = new THREE.PointsMaterial({
      vertexColors: true,
      size: pointSize
    });
    depthMesh = new THREE.Points(depthGeometry, material);
  }
  
  depthMesh.position.set(0, 1, -2);
  scene.add(depthMesh);

  depthPositions = depthGeometry.attributes.position.array;
  depthColors = depthGeometry.attributes.color.array;

  knownDepthWidth = width;
  knownDepthHeight = height;
}

/**
 * Update geometry Z offset and vertex colors each frame.
 * Each vertex's z-value is computed from the brightness of the corresponding depth pixel.
 */
function updateDepthGeometry() {
  if (!depthImgEl.complete || depthImgEl.naturalWidth === 0) return;
  if (videoEl.videoWidth === 0) return;

  const depthSlider = document.getElementById("depthslider"); 
  const scaleVal = parseFloat(depthSlider.value) || 40;

  const depthW = depthImgEl.naturalWidth;
  const depthH = depthImgEl.naturalHeight;

  // Rebuild geometry if dimensions change or if not yet created
  if (depthW !== knownDepthWidth || depthH !== knownDepthHeight || !depthMesh) {
    rebuildDepthGeometry(depthW, depthH);
  }
  if (!depthMesh) return;

  // Draw depth image into a canvas for reading pixel data
  const depthCanvas = document.createElement('canvas');
  depthCanvas.width = depthW;
  depthCanvas.height = depthH;
  const depthCtx = depthCanvas.getContext('2d');
  depthCtx.drawImage(depthImgEl, 0, 0, depthW, depthH);
  const depthData = depthCtx.getImageData(0, 0, depthW, depthH).data;

  // Draw video image into a canvas for color sampling
  const colorW = videoEl.videoWidth;
  const colorH = videoEl.videoHeight;
  const colorCanvas = document.createElement('canvas');
  colorCanvas.width = colorW;
  colorCanvas.height = colorH;
  const colorCtx = colorCanvas.getContext('2d');
  colorCtx.drawImage(videoEl, 0, 0, colorW, colorH);
  const colorData = colorCtx.getImageData(0, 0, colorW, colorH).data;

  let i3 = 0;
  for (let y = 0; y < depthH; y++) {
    for (let x = 0; x < depthW; x++) {
      const idxDepth = (y * depthW + x) * 4;
      const rD = depthData[idxDepth + 0];
      const gD = depthData[idxDepth + 1];
      const bD = depthData[idxDepth + 2];
      const gray = (rD + gD + bD) / 3;
      const zVal = (gray / 255) * scaleVal;
      depthPositions[i3 + 2] = zVal;

      // Sample color from the video feed
      const colorX = Math.floor(x * (colorW / depthW));
      const colorY = Math.floor(y * (colorH / depthH));
      const idxColor = (colorY * colorW + colorX) * 4;
      depthColors[i3 + 0] = colorData[idxColor + 0] / 255;
      depthColors[i3 + 1] = colorData[idxColor + 1] / 255;
      depthColors[i3 + 2] = colorData[idxColor + 2] / 255;
      i3 += 3;
    }
  }

  depthGeometry.attributes.position.needsUpdate = true;
  depthGeometry.attributes.color.needsUpdate = true;
}

// ---------------------------
// Render / Animation Loop
// ---------------------------
function animate() {
  renderer.setAnimationLoop(render);
}

function render() {
  updateDepthGeometry();
  intersectObjects(controller1);
  intersectObjects(controller2);
  dollyMoveXR();
  dollyMoveWASD();
  renderer.render(scene, camera);
}

// XR pointer intersection placeholders
function intersectObjects(controller) {
  let line = controller.getObjectByName("line");
  line.scale.z = 5; 
}
function onSelectStart(e) {}
function onSelectEnd(e) {}

// XR gamepad movement
function dollyMoveXR() {
  const session = renderer.xr.getSession();
  if (!session) return;

  let i = 0;
  let xrCamera = renderer.xr.getCamera(camera);
  xrCamera.getWorldDirection(cameraVector);
  cameraVector.y = 0;
  cameraVector.normalize();

  if (isIterable(session.inputSources)) {
    for (const source of session.inputSources) {
      if (!source.gamepad) continue;
      const old = prevGamePads.get(source);
      const data = {
        buttons: source.gamepad.buttons.map(b => b.value),
        axes: source.gamepad.axes.slice(0)
      };
      const controller = renderer.xr.getController(i++);

      if (old) {
        let ax2 = data.axes[2]; // turn
        let ax3 = data.axes[3]; // forward/back
        if (Math.abs(ax2) > 0.2) {
          dolly.rotateY(-THREE.Math.degToRad(ax2));
        }
        if (Math.abs(ax3) > 0.2) {
          dolly.position.x -= cameraVector.x * (ax3 * 0.1);
          dolly.position.z -= cameraVector.z * (ax3 * 0.1);
        }
      }
      prevGamePads.set(source, data);
    }
  }
}

// Desktop WASD movement
function dollyMoveWASD() {
  let forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();

  let right = new THREE.Vector3();
  right.crossVectors(forward, new THREE.Vector3(0,1,0)).normalize().negate();

  const speed = 0.05;
  if (keyStates.w) {
    dolly.position.add(forward.clone().multiplyScalar(speed));
  }
  if (keyStates.s) {
    dolly.position.add(forward.clone().multiplyScalar(-speed));
  }
  if (keyStates.a) {
    dolly.position.add(right.clone().multiplyScalar(speed));
  }
  if (keyStates.d) {
    dolly.position.add(right.clone().multiplyScalar(-speed));
  }
}

function isIterable(obj) {
  if (obj == null) return false;
  return typeof obj[Symbol.iterator] === 'function';
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
