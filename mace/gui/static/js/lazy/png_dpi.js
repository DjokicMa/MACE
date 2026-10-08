/* A PNG's resolution (plan E9): the pHYs chunk (pixels per meter = DPI /
 * 0.0254) written after IHDR, replacing one already there, so a figure keeps
 * its physical size in a word processor or a journal's checker. No
 * re-encoding: the image bytes are untouched. */
let TABLE = null;
function crcTable() {
  if (TABLE) return TABLE;
  TABLE = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    TABLE[n] = c >>> 0;
  }
  return TABLE;
}
export function crc32(bytes) {
  const t = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = t[(c ^ bytes[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function dataUrlBytes(url) {
  const b64 = url.slice(url.indexOf(",") + 1), bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesDataUrl(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return "data:image/png;base64," + btoa(s);
}
/** PNG bytes with a pHYs chunk for `dpi` (the same bytes otherwise). */
export function withPhys(png, dpi) {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!sig.every((b, i) => png[i] === b)) throw new Error("not a PNG");
  const ppm = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21);
  const dv = new DataView(chunk.buffer);
  dv.setUint32(0, 9);
  chunk.set([112, 72, 89, 115], 4);            // "pHYs"
  dv.setUint32(8, ppm); dv.setUint32(12, ppm); chunk[16] = 1;        // unit: meter
  dv.setUint32(17, crc32(chunk.subarray(4, 17)));
  // walk the chunks: keep all but an old pHYs, put ours right after IHDR
  const parts = [png.subarray(0, 8)];
  let p = 8;
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  while (p < png.length) {
    const len = view.getUint32(p), type = String.fromCharCode(...png.subarray(p + 4, p + 8)), end = p + 12 + len;
    if (type !== "pHYs") parts.push(png.subarray(p, end));
    if (type === "IHDR") parts.push(chunk);
    p = end;
  }
  const out = new Uint8Array(parts.reduce((n, x) => n + x.length, 0));
  let o = 0;
  for (const x of parts) { out.set(x, o); o += x.length; }
  return out;
}
/** A PNG data URL with its DPI set. */
export function dataUrlWithDpi(url, dpi) { return bytesDataUrl(withPhys(dataUrlBytes(url), dpi)); }
