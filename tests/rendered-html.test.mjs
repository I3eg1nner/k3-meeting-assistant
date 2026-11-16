import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

const staticRoot = new URL("../backend/static/", import.meta.url);

test("K3 production bundle contains the meeting application shell", async () => {
  const html = await readFile(new URL("index.html", staticRoot), "utf8");
  assert.match(html, /<title>K3 会议智记<\/title>/);
  assert.match(html, /<div id="root"><\/div>/);
  assert.match(html, /assets\/index-[^"]+\.js/);
});

test("K3 production bundle contains JavaScript and CSS assets", async () => {
  const assets = await readdir(new URL("assets/", staticRoot));
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});
