# WebRTC

WebRTC lets browsers establish real-time audio, video, and data connections. Signaling is used to exchange information needed to establish a connection, but signaling itself is not the media transport.

ICE candidates describe possible network paths between peers. STUN helps a peer discover its public-facing network address, while TURN can relay traffic when a direct path cannot be established.

An SFU receives media from participants and forwards selected streams to other participants. This is different from a peer-to-peer connection where participants send media directly to one another.

For a small two-person call, peer-to-peer WebRTC can be simple. As participant counts grow, an SFU can reduce the amount of upload bandwidth required from each participant.
