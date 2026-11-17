// Dependency-free .docx builder. Usage: node build_docx.js out.docx
const fs = require("fs"), zlib = require("zlib"), path = require("path");
const blocks = require("./content.js");
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function runs(text, base = {}) {            // **bold**, //italic//
  const out = []; const re = /(\*\*[^*]+\*\*|\/\/[^/]+\/\/)/g; let last = 0, m;
  const mk = (t, f) => t ? `<w:r><w:rPr>${f.b ? "<w:b/>" : ""}${f.i ? "<w:i/>" : ""}${f.sz ? `<w:sz w:val="${f.sz}"/>` : ""}</w:rPr><w:t xml:space="preserve">${esc(t)}</w:t></w:r>` : "";
  while ((m = re.exec(text))) {
    out.push(mk(text.slice(last, m.index), base));
    const tok = m[0];
    out.push(tok.startsWith("**") ? mk(tok.slice(2, -2), { ...base, b: true }) : mk(tok.slice(2, -2), { ...base, i: true }));
    last = m.index + tok.length;
  }
  out.push(mk(text.slice(last), base)); return out.join("");
}
const para = (style, inner, ppr = "") => `<w:p><w:pPr><w:pStyle w:val="${style}"/>${ppr}</w:pPr>${inner}</w:p>`;

function table(b) {
  const w = b.widths, grid = w.map(x => `<w:gridCol w:w="${x}"/>`).join("");
  const cell = (t, i, head) => `<w:tc><w:tcPr><w:tcW w:w="${w[i]}" w:type="dxa"/>${head ? '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' : ""}</w:tcPr>${para("TableText", runs(t, head ? { b: true } : {}))}</w:tc>`;
  const row = (r, head) => `<w:tr>${head ? "<w:trPr><w:tblHeader/></w:trPr>" : ""}${r.map((t, i) => cell(t, i, head)).join("")}</w:tr>`;
  return para("Caption", runs(b.caption), "<w:keepNext/>") +
    `<w:tbl><w:tblPr><w:tblW w:w="${w.reduce((a, c) => a + c, 0)}" w:type="dxa"/><w:tblBorders>${["top", "left", "bottom", "right", "insideH", "insideV"].map(s => `<w:${s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>`).join("")}</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>${grid}</w:tblGrid>${row(b.head, true)}${b.rows.map(r => row(r, false)).join("")}</w:tbl>` + para("Normal", "");
}

let body = "";
for (const b of blocks) {
  switch (b.type) {
    case "title": body += para("Title", runs(b.text)); break;
    case "authors": body += para("Authors", runs(b.text)); break;
    case "h1": body += para("Heading1", runs(b.text)); break;
    case "h2": body += para("Heading2", runs(b.text)); break;
    case "p": body += para("Normal", runs(b.text)); break;
    case "note": body += para("Note", runs(b.text), '<w:shd w:val="clear" w:color="auto" w:fill="FFF2CC"/><w:pBdr><w:top w:val="single" w:sz="4" w:space="4" w:color="BF9000"/><w:left w:val="single" w:sz="4" w:space="4" w:color="BF9000"/><w:bottom w:val="single" w:sz="4" w:space="4" w:color="BF9000"/><w:right w:val="single" w:sz="4" w:space="4" w:color="BF9000"/></w:pBdr>'); break;
    case "eq": body += para("Equation", `<w:r><w:t xml:space="preserve">${esc(b.text)}</w:t></w:r><w:r><w:tab/><w:t>${esc(b.num || "")}</w:t></w:r>`, '<w:tabs><w:tab w:val="right" w:pos="9360"/></w:tabs>'); break;
    case "bullets": for (const t of b.text) body += para("Bullet", `<w:r><w:t xml:space="preserve">•\t</w:t></w:r>` + runs(t)); break;
    case "table": body += table(b); break;
    case "ref": body += para("Reference", runs(b.text)); break;
  }
}
const sect = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>';
const W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
const document = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document ${W}><w:body>${body}${sect}</w:body></w:document>`;
const sty = (id, name, ppr, rpr, extra = "") => `<w:style w:type="paragraph" ${id === "Normal" ? 'w:default="1" ' : ""}w:styleId="${id}"><w:name w:val="${name}"/>${extra}<w:pPr>${ppr}</w:pPr><w:rPr>${rpr}</w:rPr></w:style>`;
const F = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>';
const styles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles ${W}><w:docDefaults><w:rPrDefault><w:rPr>${F}<w:sz w:val="22"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault></w:docDefaults>` +
  sty("Normal", "Normal", '<w:spacing w:after="120" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>', "") +
  sty("Title", "Title", '<w:spacing w:before="0" w:after="160"/><w:jc w:val="center"/>', `${F}<w:b/><w:sz w:val="36"/>`, '<w:basedOn w:val="Normal"/>') +
  sty("Authors", "Authors", '<w:spacing w:after="200"/><w:jc w:val="center"/>', `${F}<w:sz w:val="22"/>`, '<w:basedOn w:val="Normal"/>') +
  sty("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="280" w:after="120"/><w:jc w:val="left"/><w:outlineLvl w:val="0"/>', `${F}<w:b/><w:sz w:val="28"/>`, '<w:basedOn w:val="Normal"/>') +
  sty("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="200" w:after="80"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/>', `${F}<w:b/><w:i/><w:sz w:val="24"/>`, '<w:basedOn w:val="Normal"/>') +
  sty("Equation", "Equation", '<w:spacing w:before="80" w:after="120"/><w:jc w:val="left"/>', `<w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/>`, '<w:basedOn w:val="Normal"/>') +
  sty("Bullet", "Bullet", '<w:spacing w:after="80"/><w:ind w:left="540" w:hanging="270"/>', "", '<w:basedOn w:val="Normal"/>') +
  sty("Note", "Note", '<w:spacing w:before="120" w:after="160"/><w:jc w:val="left"/>', '<w:sz w:val="20"/>', '<w:basedOn w:val="Normal"/>') +
  sty("Caption", "caption", '<w:spacing w:before="160" w:after="80"/><w:jc w:val="left"/>', '<w:b/><w:sz w:val="20"/>', '<w:basedOn w:val="Normal"/>') +
  sty("TableText", "Table Text", '<w:spacing w:after="20" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/>', '<w:sz w:val="19"/>', '<w:basedOn w:val="Normal"/>') +
  sty("Reference", "Reference", '<w:spacing w:after="60"/><w:ind w:left="540" w:hanging="540"/><w:jc w:val="left"/>', '<w:sz w:val="20"/>', '<w:basedOn w:val="Normal"/>') + "</w:styles>";
const files = {
  "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>',
  "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
  "word/_rels/document.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
  "word/document.xml": document, "word/styles.xml": styles
};
// minimal zip writer (deflate)
const crcT = (() => { const t = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = b => { let c = 0xFFFFFFFF; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
const parts = [], cen = []; let off = 0;
for (const [name, txt] of Object.entries(files)) {
  const data = Buffer.from(txt, "utf8"), comp = zlib.deflateRawSync(data), nm = Buffer.from(name), crc = crc32(data);
  const lh = Buffer.alloc(30); lh.writeUInt32LE(0x04034b50, 0); lh.writeUInt16LE(20, 4); lh.writeUInt16LE(0, 6); lh.writeUInt16LE(8, 8); lh.writeUInt32LE(0x21, 10); lh.writeUInt32LE(crc, 14); lh.writeUInt32LE(comp.length, 18); lh.writeUInt32LE(data.length, 22); lh.writeUInt16LE(nm.length, 26);
  parts.push(lh, nm, comp);
  const ch = Buffer.alloc(46); ch.writeUInt32LE(0x02014b50, 0); ch.writeUInt16LE(20, 4); ch.writeUInt16LE(20, 6); ch.writeUInt16LE(8, 10); ch.writeUInt32LE(0x21, 12); ch.writeUInt32LE(crc, 16); ch.writeUInt32LE(comp.length, 20); ch.writeUInt32LE(data.length, 24); ch.writeUInt16LE(nm.length, 28); ch.writeUInt32LE(off, 42);
  cen.push(ch, nm); off += 30 + nm.length + comp.length;
}
const cenBuf = Buffer.concat(cen), end = Buffer.alloc(22);
end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(Object.keys(files).length, 8); end.writeUInt16LE(Object.keys(files).length, 10); end.writeUInt32LE(cenBuf.length, 12); end.writeUInt32LE(off, 16);
fs.writeFileSync(process.argv[2], Buffer.concat([...parts, cenBuf, end]));
const words = blocks.map(b => Array.isArray(b.text) ? b.text.join(" ") : (b.text || "")).join(" ").split(/\s+/).length;
console.log("wrote", process.argv[2], "~" + words + " words");
