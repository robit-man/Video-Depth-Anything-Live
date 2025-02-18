const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http, {
  cors: {
    origin: "*", // For production, restrict to your client's URL.
    methods: ["GET", "POST"]
  }
});

let inferenceSocket = null;
let clientSockets = {};

io.on('connection', (socket) => {
  console.log(`Socket connected: ${socket.id}`);

  socket.on('register_inference', () => {
    inferenceSocket = socket;
    console.log(`Inference server registered: ${socket.id}`);
  });

  socket.on('register_client', () => {
    clientSockets[socket.id] = socket;
    console.log(`Client registered: ${socket.id}`);
  });

  // Relay WebRTC signaling events.
  socket.on('webrtc_offer', (data) => {
    console.log(`Received WebRTC offer from client ${socket.id}`);
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('webrtc_offer', data);
    }
  });

  socket.on('webrtc_answer', (data) => {
    console.log(`Received WebRTC answer for client ${data.client_id}`);
    if (data.client_id && clientSockets[data.client_id]) {
      clientSockets[data.client_id].emit('webrtc_answer', { sdp: data.sdp });
    }
  });

  socket.on('webrtc_candidate', (data) => {
    console.log(`ICE candidate from ${socket.id} for target ${data.target}`);
    if (data.target === "inference" && inferenceSocket) {
      data.client_id = socket.id;
      inferenceSocket.emit("webrtc_candidate", data);
    } else if (data.target === "client" && data.client_id && clientSockets[data.client_id]) {
      clientSockets[data.client_id].emit("webrtc_candidate", { candidate: data.candidate });
    }
  });

  // Relay fallback events.
  socket.on('video_frame', (data) => {
    console.log(`Video frame received from client ${socket.id}`);
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('video_frame', data);
    }
  });

  socket.on('frame', (data) => {
    console.log(`Frame received from client ${socket.id}`);
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('frame', data);
    }
  });

  socket.on('depth_frame', (data) => {
    console.log(`Depth frame received for client ${data.client_id}`);
    const clientId = data.client_id;
    if (clientId && clientSockets[clientId]) {
      clientSockets[clientId].emit('depth_frame', data);
    } else {
      Object.values(clientSockets).forEach((client) => {
        client.emit('depth_frame', data);
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
