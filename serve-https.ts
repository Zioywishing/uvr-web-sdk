import { serve } from "bun";
import { join } from "path";
import { networkInterfaces } from "os";

/**
 * 本脚本用于使用 Bun 原生 API 托管静态文件并通过 HTTPS 访问。
 * 默认监听端口为 4433。
 */

const PORT = 4433;
const ROOT_DIR = process.cwd();
const DIST_DIR = join(ROOT_DIR, "demo-html/dist");
const KEY_PATH = join(ROOT_DIR, "certs/key.pem");
const CERT_PATH = join(ROOT_DIR, "certs/cert.pem");

// 获取本机所有有效的 IPv4 地址
function getLocalIPs(): string[] {
  const ips: string[] = [];
  const interfaces = networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    const netInters = interfaces[name];
    if (!netInters) continue;
    for (const net of netInters) {
      // 过滤 IPv4 且非回环地址
      if (net.family === "IPv4" && !net.internal) {
        ips.push(net.address);
      }
    }
  }
  return ips;
}

console.log("-----------------------------------------");
console.log(`正在启动 HTTPS 静态文件服务...`);
console.log(`托管目录: ${DIST_DIR}`);
console.log(`证书私钥: ${KEY_PATH}`);
console.log(`证书文件: ${CERT_PATH}`);
console.log("-----------------------------------------");

const server = serve({
  port: PORT,
  hostname: "0.0.0.0",
  tls: {
    key: Bun.file(KEY_PATH),
    cert: Bun.file(CERT_PATH),
  },
  async fetch(req) {
    const url = new URL(req.url);
    let path = url.pathname;

    // 默认跳转到 index.html
    if (path === "/") {
      path = "/index.html";
    }

    const filePath = join(DIST_DIR, path);
    const file = Bun.file(filePath);

    if (await file.exists()) {
      return new Response(file, {
        headers: {
          "Cross-Origin-Opener-Policy": "same-origin",
          "Cross-Origin-Embedder-Policy": "require-corp",
        },
      });
    }

    // 如果找不到文件，尝试返回 index.html (支持 SPA 路由)
    const indexFile = Bun.file(join(DIST_DIR, "index.html"));
    if (await indexFile.exists()) {
      return new Response(indexFile, {
        headers: {
          "Cross-Origin-Opener-Policy": "same-origin",
          "Cross-Origin-Embedder-Policy": "require-corp",
        },
      });
    }

    return new Response("未找到文件", { status: 404 });
  },
  error(error) {
    console.error("服务器错误:", error);
    return new Response(`服务器内部错误: ${error.message}`, { status: 500 });
  },
});

console.log(`服务已成功启动！`);
console.log(`本地访问: https://localhost:${server.port}`);

const ips = getLocalIPs();
if (ips.length > 0) {
  ips.forEach((ip) => {
    console.log(`网络访问: https://${ip}:${server.port}`);
  });
} else {
  console.log(`网络访问: https://${server.hostname}:${server.port}`);
}

console.log("按 Ctrl+C 停止服务");
console.log("-----------------------------------------");
