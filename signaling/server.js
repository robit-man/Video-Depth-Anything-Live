// server.js
const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http, {
  cors: {
    origin: "*", // For production, restrict to your client's URL.
    methods: ["GET", "POST"]
  }
});

// We'll support one inference server and multiple clients.
let inferenceSocket = null;
let clientSockets = {}; // Map: socket.id => socket

io.on('connection', (socket) => {
  console.log(`Socket connected: ${socket.id}`);

  // Inference server registration.
  socket.on('register_inference', () => {
    inferenceSocket = socket;
    console.log(`Inference server registered: ${socket.id}`);
  });

  // Client registration.
  socket.on('register_client', () => {
    clientSockets[socket.id] = socket;
    console.log(`Client registered: ${socket.id}`);
  });

  // WebRTC signaling: client sends offer.
  socket.on('webrtc_offer', (data) => {
    console.log(`Received WebRTC offer from client ${socket.id}`);
    // Attach the client ID.
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('webrtc_offer', data);
    }
  });

  // Inference server sends answer.
  socket.on('webrtc_answer', (data) => {
    const clientId = data.client_id;
    console.log(`Received WebRTC answer for client ${clientId}`);
    if (clientId && clientSockets[clientId]) {
      clientSockets[clientId].emit('webrtc_answer', { sdp: data.sdp });
    }
  });

  // Relay ICE candidates.
  socket.on('webrtc_candidate', (data) => {
    console.log(`Received ICE candidate from ${socket.id} for target ${data.target}`);
    if (data.target === "inference" && inferenceSocket) {
      // From client to inference server.
      data.client_id = socket.id;
      inferenceSocket.emit("webrtc_candidate", data);
    } else if (data.target === "client" && data.client_id && clientSockets[data.client_id]) {
      clientSockets[data.client_id].emit("webrtc_candidate", { candidate: data.candidate });
    }
  });

  // Fallback: Relay video frames via Socket.IO.
  socket.on('video_frame', (data) => {
    console.log(`Video frame received from client ${socket.id}`);
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit("video_frame", data);
    }
  });

  // Inference server sends back processed depth frames.
  socket.on('depth_frame', (data) => {
    console.log(`Depth frame received for client ${data.client_id}`);
    const clientId = data.client_id;
    if (clientId && clientSockets[clientId]) {
      clientSockets[clientId].emit("depth_frame", data);
    } else {
      // Fallback broadcast.
      Object.values(clientSockets).forEach((client) => {
        client.emit("depth_frame", data);
      });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Socket disconnected: ${socket.id}`);
    if (socket === inferenceSocket) {
      inferenceSocket = null;
      console.log('Inference server disconnected.');
    }
    delete clientSockets[socket.id];
  });
});

http.listen(process.env.PORT || 3000, () => {
  console.log(`Signaling server listening on port ${process.env.PORT || 3000}`);
});
