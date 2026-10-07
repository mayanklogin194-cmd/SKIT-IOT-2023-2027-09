// Same-origin in production (Flask serves the build); proxied to :5000 by Vite in dev.
const BASE = import.meta.env.VITE_API_BASE || "";

export async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

export async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}
