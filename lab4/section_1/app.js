const http = require("http");
const path = require("path");
const fs = require("fs");
const { URL } = require("url");
const { Server } = require("socket.io");
const { io: iocli } = require("socket.io-client");

const PORTS = { DASH: 3000, A: 8001, B: 8002 };
const LOG_CAP = 150;
const MESSAGE_CAP = 500;

const srv = {
  A: { running: false, httpServer: null, io: null, peerClient: null },
  B: { running: false, httpServer: null, io: null, peerClient: null },
};

const clients = new Map();
const logs = [];
const messages = [];

function timestamp() {
  return new Date().toLocaleTimeString("mn");
}

function pushLog(type, text, extra = {}) {
  logs.unshift({ type, text, ts: timestamp(), ...extra });
  if (logs.length > LOG_CAP) logs.pop();
}

function pushMessage(entry) {
  messages.unshift({ ...entry, ts: timestamp() });
  if (messages.length > MESSAGE_CAP) messages.pop();
}

function clientsOn(name) {
  let n = 0;
  for (const c of clients.values()) {
    if (c.connectedServer === name) n++;
  }
  return n;
}

function getState() {
  return {
    servers: {
      A: { running: srv.A.running, clients: clientsOn("A") },
      B: { running: srv.B.running, clients: clientsOn("B") },
    },
    clients: Array.from(clients.values()).map((c) => ({
      num: c.num,
      assignedServer: c.assignedServer,
      connectedServer: c.connectedServer,
      status: c.status,
    })),
    logs,
    messages,
  };
}

function sendJson(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}

function apiError(res, message, status = 400) {
  sendJson(res, status, { ok: false, error: message });
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > 1e6) {
        req.destroy();
        reject(new Error("Payload too large"));
      }
    });
    req.on("end", () => {
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("Invalid JSON"));
      }
    });
    req.on("error", reject);
  });
}

function closePeerClient(s) {
  if (!s.peerClient) return;
  s.peerClient.removeAllListeners();
  s.peerClient.disconnect();
  s.peerClient = null;
}

function ensurePeerLink(name) {
  const s = srv[name];
  const peerName = name === "A" ? "B" : "A";
  const peer = srv[peerName];

  if (!s.running) return;
  if (!peer.running) return closePeerClient(s);
  if (s.peerClient && s.peerClient.connected) return;

  closePeerClient(s);

  const peerClient = iocli(`http://localhost:${PORTS[peerName]}`, {
    query: { role: "peer", from: name },
    reconnection: true,
    reconnectionDelay: 1000,
    timeout: 2000,
  });
  s.peerClient = peerClient;

  peerClient.on("connect", () => {
    pushLog("server", `Server ${name} linked to Server ${peerName} via socket`);
  });
}

function startServer(name) {
  const s = srv[name];
  if (s.running) return;

  s.httpServer = http.createServer();
  s.io = new Server(s.httpServer, { cors: { origin: "*" } });

  s.io.on("connection", (socket) => {
    if (socket.handshake.query.role === "peer") {
      // Step 4: peer server receives forwarded message and rebroadcasts locally.
      socket.on("peer-message", ({ from, message, originServer } = {}) => {
        s.io.emit("message", {
          from,
          message,
          server: originServer,
          forwarded: true,
        });
      });
      return;
    }

    const { clientId } = socket.handshake.query;
    if (!clientId) return;

    const num = parseInt(clientId.replace("client-", ""));
    socket.join(`client-${num}`);

    const c = clients.get(num);
    if (c) {
      c.connectedServer = name;
      c.status = "connected";
    }
    pushLog("join", `Client ${num} connected to Server ${name}`);

    // Step 1: connected client sends message to its current server.
    socket.on("message", ({ message }) => {
      const recipients = Array.from(clients.values())
        .filter((other) => other.status === "connected" && other.num !== num)
        .map((other) => other.num);

      // Step 2: server broadcasts to its own connected clients.
      s.io.emit("message", { from: num, message, server: name });

      // Step 3: server forwards message to peer server through socket link.
      if (s.peerClient && s.peerClient.connected) {
        s.peerClient.emit("peer-message", {
          from: num,
          message,
          originServer: name,
        });
      }

      pushLog("message", `[Client ${num}] ${message}`, {
        from: num,
        message,
        server: name,
        recipients,
      });
      pushMessage({ from: num, message, server: name, recipients });
    });

    socket.on("disconnect", () => {
      const c = clients.get(num);
      if (c && c.connectedServer === name) {
        c.connectedServer = null;
        c.status = "reconnecting";
      }
      pushLog("leave", `Client ${num} disconnected from Server ${name}`);
    });
  });

  s.httpServer.listen(PORTS[name], () => {
    s.running = true;
    pushLog("server", `Server ${name} started on :${PORTS[name]}`);

    ensurePeerLink(name);
    ensurePeerLink(name === "A" ? "B" : "A");

    clients.forEach((c) => {
      if (c.assignedServer === name && c.connectedServer !== name) {
        connectClient(c.num, name);
      }
    });
  });
}

function stopServer(name) {
  const s = srv[name];
  if (!s.running) return;

  clients.forEach((c) => {
    if (c.connectedServer === name) {
      c.connectedServer = null;
      c.status = "reconnecting";
    }
  });

  closePeerClient(s);
  s.io.disconnectSockets(true);
  s.httpServer.close(() => {
    s.running = false;
    s.io = null;
    s.httpServer = null;
    pushLog("server", `Server ${name} stopped`);

    // Fault tolerance: redirect assigned clients to the other server.
    const fallback = name === "A" ? "B" : "A";
    clients.forEach((c) => {
      if (c.assignedServer !== name) return;
      if (srv[fallback].running) {
        pushLog(
          "fault",
          `Client ${c.num}: Server ${name} down → failover to Server ${fallback}`,
        );
        connectClient(c.num, fallback);
      } else {
        c.status = "no-server";
      }
    });
  });
}

function connectClient(num, targetServer) {
  const c = clients.get(num);
  if (!c) return;

  if (!srv[targetServer].running) {
    const other = targetServer === "A" ? "B" : "A";
    if (srv[other].running) return connectClient(num, other);
    c.status = "no-server";
    return;
  }

  if (c.socket) {
    c.socket.removeAllListeners();
    c.socket.disconnect();
    c.socket = null;
  }

  c.status = "connecting";

  const socket = iocli(`http://localhost:${PORTS[targetServer]}`, {
    query: { clientId: `client-${num}` },
    reconnection: true,
    reconnectionAttempts: 3,
    reconnectionDelay: 800,
    timeout: 2000,
  });
  c.socket = socket;

  socket.on("connect", () => {
    c.connectedServer = targetServer;
    c.status = "connected";
  });

  socket.io.on("reconnect_failed", () => {
    const fallback = targetServer === "A" ? "B" : "A";
    if (srv[fallback].running) {
      pushLog(
        "fault",
        `Client ${num}: Server ${targetServer} unreachable → failover to Server ${fallback}`,
      );
      connectClient(num, fallback);
    } else {
      c.status = "no-server";
    }
  });

  socket.on("disconnect", () => {
    if (c.connectedServer === targetServer) {
      c.connectedServer = null;
      c.status = "reconnecting";
    }
  });
}

function addClient(num) {
  if (isNaN(num) || num < 1 || num > 99) {
    return { ok: false, error: "Клиентийн дугаар 1–99 байх ёстой" };
  }
  if (clients.has(num)) {
    return { ok: false, error: `Client ${num} аль хэдийн байна` };
  }

  const assignedServer = num % 2 === 0 ? "A" : "B";
  clients.set(num, {
    num,
    assignedServer,
    connectedServer: null,
    socket: null,
    status: "connecting",
  });

  pushLog(
    "join",
    `Client ${num} нэмэгдлээ (хуваарилагдсан: Server ${assignedServer})`,
  );
  connectClient(num, assignedServer);
  return { ok: true };
}

function removeClient(num) {
  const c = clients.get(num);
  if (!c) return { ok: false, error: `Client ${num} олдсонгүй` };
  if (c.socket) {
    c.socket.removeAllListeners();
    c.socket.disconnect();
  }
  clients.delete(num);
  pushLog("leave", `Client ${num} устгагдлаа`);
  return { ok: true };
}

function sendMessage(fromNum, message) {
  const c = clients.get(fromNum);
  if (!c) return { ok: false, error: `Client ${fromNum} олдсонгүй` };
  if (!c.socket || !c.socket.connected) {
    return {
      ok: false,
      error: `Client ${fromNum} сервертэй холбогдоогүй байна`,
    };
  }
  c.socket.emit("message", { message });
  return { ok: true };
}

const handlers = {
  "POST /api/server/toggle": (body) => {
    const name = body?.name;
    if (name !== "A" && name !== "B") {
      return { ok: false, error: "Server нэр буруу байна" };
    }
    srv[name].running ? stopServer(name) : startServer(name);
    return { ok: true };
  },
  "POST /api/clients/add": (body) => addClient(parseInt(body?.num)),
  "POST /api/clients/remove": (body) => removeClient(parseInt(body?.num)),
  "POST /api/messages/send": (body) =>
    sendMessage(parseInt(body?.from), `${body?.message || ""}`),
  "POST /api/logs/clear": () => {
    logs.length = 0;
    return { ok: true };
  },
  "POST /api/messages/clear": () => {
    messages.length = 0;
    return { ok: true };
  },
};

const dashHttp = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const route = `${req.method} ${url.pathname}`;

  if (route === "GET /") {
    const file = path.join(__dirname, "public", "index.html");
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    fs.createReadStream(file).pipe(res);
    return;
  }

  if (route === "GET /api/state") {
    return sendJson(res, 200, { ok: true, state: getState() });
  }

  const handler = handlers[route];
  if (!handler) {
    res.writeHead(404);
    res.end();
    return;
  }

  try {
    const body = await parseBody(req);
    const result = handler(body);
    if (!result.ok) return apiError(res, result.error);
    sendJson(res, 200, { ok: true, state: getState() });
  } catch (err) {
    apiError(
      res,
      err.message === "Invalid JSON" ? "JSON буруу байна" : "Алдаа гарлаа",
    );
  }
});

dashHttp.listen(PORTS.DASH, () => {
  console.log("\n╔══════════════════════════════════════════╗");
  console.log("║   Distributed System Dashboard           ║");
  console.log(`║   http://localhost:${PORTS.DASH}                 ║`);
  console.log("╚══════════════════════════════════════════╝\n");
  startServer("A");
  startServer("B");
});
