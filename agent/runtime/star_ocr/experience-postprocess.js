import { rgbToOpenCvHsv } from "./experience-geometry.js";


export const EXPERIENCE_CANONICAL                                                            = {
  orange: "橙星曜",
  purple: "紫星曜",
  white: "白星曜",
};

const ORDER                                           = { orange: 0, purple: 1, white: 2 };
const HUE_TARGETS                                           = { orange: 10, purple: 130, white: 100 };

function median(values          )         {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]  : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

export function classifyExperienceKind(image           , icon      )                                                                          {
  const left = Math.max(0, icon.x + Math.trunc(icon.width * 0.22));
  const top = Math.max(0, icon.y + Math.trunc(icon.height * 0.22));
  const width = Math.max(1, Math.trunc(icon.width * 0.56));
  const height = Math.max(1, Math.trunc(icon.height * 0.56));
  const hues           = [];
  for (let y = top; y < Math.min(image.height, top + height); y += 1) {
    for (let x = left; x < Math.min(image.width, left + width); x += 1) {
      const at = (y * image.width + x) * 4;
      const [hue, saturation] = rgbToOpenCvHsv(image.data[at] ?? 0, image.data[at + 1] ?? 0, image.data[at + 2] ?? 0);
      if (saturation > 70) hues.push(hue);
    }
  }
  if (!hues.length) return { kind: null, confidence: 0, hue: null };
  const hue = median(hues);
  const kinds = Object.keys(HUE_TARGETS)                    ;
  const kind = kinds.reduce((best, candidate) => Math.abs(hue - HUE_TARGETS[candidate]) < Math.abs(hue - HUE_TARGETS[best]) ? candidate : best, kinds[0] );
  return { kind, confidence: Math.max(0, 1 - Math.abs(hue - HUE_TARGETS[kind]) / 35), hue };
}

export function parseExperienceCount(text        )                {
  const match = /^\d{1,6}$/u.exec(text.trim());
  return match ? Number(match[0]) : null;
}

export function resolveExperienceCount(candidates                )                                                                               {
  const valid = candidates
    .map((candidate) => ({ ...candidate, count: parseExperienceCount(candidate.text) }))
    .filter((candidate)                                                => candidate.count != null)
    .sort((left, right) => right.confidence - left.confidence);
  const best = valid[0];
  if (best) return { raw: best.text, count: best.count, confidence: best.confidence, reasons: [] };
  return {
    raw: [...candidates].sort((left, right) => right.confidence - left.confidence)[0]?.text ?? "",
    count: null,
    confidence: 0,
    reasons: ["experience_count_unparsed"],
  };
}

export function experienceStatus(completeness        , kind                       , count               )                             {
  if (completeness !== "complete") return "excluded_partial";
  return kind != null && count != null ? "accepted" : "needs_review";
}

export function aggregateExperience(
  results                    ,
  options                                                    ,
)                      {
  const warnings           = [];
  if (!results.some((item) => item.status !== "excluded_partial")) warnings.push("experience_icons_not_found");
  const classified = results.filter((item)                                                      => item.status !== "excluded_partial" && item.kind != null);
  const orderedKinds = classified.map((item) => item.kind);
  if (orderedKinds.some((kind, index) => index > 0 && ORDER[kind] < ORDER[orderedKinds[index - 1] ])) warnings.push("experience_order_conflict");
  if (results.some((item) => item.status !== "excluded_partial" && item.kind == null)) warnings.push("experience_icon_unclassified");
  for (const item of classified) if (item.count == null) warnings.push(`experience_count_unparsed:${item.kind}`);
  if (!options.selectedTab) warnings.push("experience_tab_selection_unverified");
  if (options.viewportCropped) warnings.push("experience_viewport_cropped");
  const complete = options.selectedTab && !options.viewportCropped && warnings.length === 0;
  const byKind = new Map                                  ();
  for (const item of classified) {
    const previous = byKind.get(item.kind);
    const score = Math.min(item.kindConfidence, item.countConfidence);
    const previousScore = previous ? Math.min(previous.kindConfidence, previous.countConfidence) : -1;
    if (!previous || score > previousScore) byKind.set(item.kind, item);
  }
  const field = (kind                )                => byKind.get(kind)?.count ?? (complete ? 0 : null);
  return { orangeCount: field("orange"), purpleCount: field("purple"), whiteCount: field("white"), complete, warnings };
}
