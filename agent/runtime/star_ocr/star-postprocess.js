








function round(value        )         { return Math.round(value * 100) / 100; }

export function cleanName(value        )         {
  return value.replace(/[\s_级級]+/gu, "").replace(/[^\p{L}\p{N}]/gu, "");
}

export function resolveCatalogName(
  candidates                ,
  allowedNames                   ,
  aliases                                  ,
)                 {
  const cleaned = candidates.map((item) => ({ ...item, text: cleanName(item.text) })).filter((item) => item.text.length > 0);
  if (!cleaned.length) return { raw: "", normalized: null, confidence: 0, reasons: ["name_ocr_empty"] };
  const best = [...cleaned].sort((a, b) => b.confidence - a.confidence)[0] ;
  const normalized = aliases[best.text] ?? best.text;
  if (allowedNames.includes(normalized)) {
    const agreement = cleaned.filter((item) => (aliases[item.text] ?? item.text) === normalized).length;
    return { raw: best.text, normalized, confidence: round(Math.min(0.99, best.confidence * 0.85 + Math.min(agreement, 3) * 0.05)), reasons: [] };
  }
  return { raw: best.text, normalized: null, confidence: round(best.confidence * 0.5), reasons: ["name_unknown"] };
}

export function parseLevel(value        )                {
  if (value.includes("-") || /[负負]/u.test(value)) return null;
  const matches = value.replace(/[Oo]/gu, "0").match(/\d+/gu) ?? [];
  if (matches.length !== 1) return null;
  const level = Number(matches[0]);
  return Number.isInteger(level) && level >= 1 && level <= 60 ? level : null;
}

export function resolveLevel(candidates                )                                                                               {
  const rawBest = [...candidates].sort((a, b) => b.confidence - a.confidence)[0];
  const valid = candidates.map((item) => ({ ...item, level: parseLevel(item.text) })).filter((item)                                           => item.level != null);
  if (!valid.length) return { raw: rawBest?.text ?? "", level: null, confidence: 0, reasons: ["level_unknown"] };
  const groups = new Map                                                 ();
  valid.forEach((item) => groups.set(item.level, [...(groups.get(item.level) ?? []), item]));
  const ranked = [...groups.entries()].map(([level, items]) => ({ level, items, weight: items.reduce((sum, item) => sum + item.confidence + (/[级級]/u.test(item.text) ? 0.22 : 0), 0) })).sort((a, b) => b.weight - a.weight);
  const winner = ranked[0] ;
  const best = [...winner.items].sort((a, b) => b.confidence - a.confidence)[0] ;
  const completeHigh = /[级級]/u.test(best.text) && best.confidence >= 0.85;
  if (ranked.length > 1 && !completeHigh && winner.weight - ranked[1] .weight < 0.15) {
    return { raw: best.text, level: null, confidence: 0, reasons: ["level_strategy_conflict", `level_candidates:${ranked.map((item) => item.level).join("/")}`] };
  }
  return {
    raw: best.text, level: winner.level,
    confidence: round(Math.min(0.99, best.confidence * 0.82 + Math.min(winner.items.length, 3) * 0.06)),
    reasons: ranked.length === 1 ? [] : [`level_weighted_consensus:${ranked.map((item) => item.level).join("/")}`],
  };
}

export function classifyStarResultStatus(completeness              , normalizedName               , level               )                                                   {
  if (completeness !== "complete") return "excluded_partial";
  return normalizedName && level != null ? "accepted" : "needs_review";
}
