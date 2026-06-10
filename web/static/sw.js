// Minimal service worker: network-first passthrough. Exists so the PWA is
// installable (Add to Home Screen / Install app); the game is live-only —
// nothing useful to serve offline, so no caching.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(clients.claim()));
self.addEventListener("fetch", () => {});  // passthrough
