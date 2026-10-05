# WebSockets

A WebSocket connection provides a long-lived, bidirectional communication channel between a client and server. Unlike normal HTTP request-response traffic, either side can send messages after the connection is established.

WebSockets are useful for things such as chat, presence updates, live notifications, and signaling for real-time applications.

A server can keep track of which clients are connected and broadcast an event to a group of clients. In a multi-instance deployment, that local connection state cannot automatically be shared by every server instance.

A heartbeat or ping mechanism can help detect dead connections. When a connection closes, the server should clean up its presence or subscription state.
