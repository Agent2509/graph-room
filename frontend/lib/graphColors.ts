/** Single source of truth for graph colours (canvas + legend). */
export const TYPE_COLORS: Record<string, string> = {
  PERSON: "#e0a458",
  ORGANIZATION: "#8fbf9b",
  TECHNOLOGY: "#7fa6c9",
  PRODUCT: "#7fa6c9",
  CONCEPT: "#b9b3a3",
  LOCATION: "#c4693f",
  EVENT: "#c58fb5",
  DOCUMENT: "#5f7f8c",
};

export const COMMUNITY_COLORS = [
  "#7fa6c9",
  "#8fbf9b",
  "#e0a458",
  "#c58fb5",
  "#8aa4d6",
  "#d69a7a",
  "#a3c9a8",
  "#c9b47f",
  "#9bb8c4",
  "#b9a3c9",
];

export const colorForType = (type: string) => TYPE_COLORS[type] ?? TYPE_COLORS.CONCEPT;
