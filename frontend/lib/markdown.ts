/** Minimal, escaped markdown: paragraphs, bullets, bold, inline code. */
const escape = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const inline = (text: string) =>
  text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

export function toHtml(text: string): string {
  return escape(text)
    .split(/\n{2,}/)
    .filter((block) => block.trim())
    .map((block) => {
      const lines = block.split("\n").filter((line) => line.trim());
      if (lines.length && lines.every((line) => /^\s*[-*]\s+/.test(line))) {
        const items = lines
          .map((line) => `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }
      return `<p>${inline(block).replace(/\n/g, "<br/>")}</p>`;
    })
    .join("");
}
