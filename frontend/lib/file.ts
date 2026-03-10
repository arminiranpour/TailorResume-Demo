import { strFromU8, unzipSync } from "fflate";
import { getFileExtension } from "./validation";

type ResumeInput = {
  text: string;
  docxBase64?: string;
};

export async function readResumeInput(file: File): Promise<ResumeInput> {
  const extension = getFileExtension(file.name);
  if (extension === ".txt") {
    return { text: await file.text() };
  }

  if (extension === ".docx") {
    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const base64 = bufferToBase64(bytes);
    const resumeText = extractDocxText(bytes);
    return { text: resumeText, docxBase64: base64 };
  }

  return { text: await file.text() };
}

function bufferToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function extractDocxText(bytes: Uint8Array): string {
  try {
    const files = unzipSync(bytes);
    const docXml = files["word/document.xml"];
    if (!docXml) {
      return "";
    }
    const xmlText = strFromU8(docXml);
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlText, "application/xml");
    const paragraphs = Array.from(doc.getElementsByTagName("w:p"));
    const lines = paragraphs.map((paragraph) => {
      const nodes = Array.from(paragraph.getElementsByTagName("w:t"));
      return nodes.map((node) => node.textContent ?? "").join("");
    });
    return lines.join("\n").trim();
  } catch {
    return "";
  }
}

export function buildObjectUrlFromDocx(result: {
  blob?: Blob;
  base64?: string;
  bytes?: number[];
  mimeType?: string;
}): string | null {
  if (result.blob) {
    return URL.createObjectURL(result.blob);
  }

  if (result.bytes && result.bytes.length > 0) {
    const blob = new Blob([new Uint8Array(result.bytes)], {
      type: result.mimeType ?? "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });
    return URL.createObjectURL(blob);
  }

  if (result.base64) {
    const binary = atob(result.base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], {
      type: result.mimeType ?? "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });
    return URL.createObjectURL(blob);
  }

  return null;
}
