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
let clientSockets = {}; // Use an object mapping socket.id -> socket

io.on('connection', (socket) => {
  console.log(`Socket connected: ${socket.id}`);

  // When an inference server connects, register it.
  socket.on('register_inference', () => {
    inferenceSocket = socket;
    console.log(`Inference server registered: ${socket.id}`);
  });

  // When a client connects, register it.
  socket.on('register_client', () => {
    clientSockets[socket.id] = socket;
    console.log(`Client registered: ${socket.id}`);
  });

  // Relay video frame from a client to the inference server.
  socket.on('video_frame', (data) => {
    console.log(`Video frame received from client ${socket.id}`);
    // Attach the client id to the data.
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('video_frame', data);
    } else {
      console.log('No inference server connected; cannot forward video frame.');
    }
  });

  // Also relay frames if the event name is "frame"
  socket.on('frame', (data) => {
    console.log(`Frame received from client ${socket.id}`);
    data.client_id = socket.id;
    if (inferenceSocket) {
      inferenceSocket.emit('frame', data);
    }
  });

  // Relay the processed depth frame from the inference server back to the originating client.
  socket.on('depth_frame', (data) => {
    console.log(`Depth frame received from inference server with client_id ${data.client_id}`);
    const clientId = data.client_id;
    if (clientId && clientSockets[clientId]) {
      clientSockets[clientId].emit('depth_frame', data);
    } else {
      console.log(`Client ${clientId} not found; broadcasting depth frame to all clients.`);
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
