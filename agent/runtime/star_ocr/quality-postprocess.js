

import { rgbToOpenCvHsv } from "./experience-geometry.js";

export const QUALITY_HUES                                                  = {
  橙: 12,
  紫: 142,
  蓝: 106,
  绿: 61,
};















export function qualityRectForCard(card               )       {
  return { ...card.cardRect };
}

function hueDistance(left        , right        )         {
  const distance = Math.abs(left - right);
  return Math.min(distance, 180 - distance);
}

function median(values                   )         {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

/** Direct browser equivalent of Python quality_recognizer.classify_quality_pixels. */
export function classifyQualityPixels(pixels                     )                     {
  if (!pixels.length) {
    return { quality: null, confidence: 0, evidence: "empty_relative_quality_region", warnings: ["quality_unknown"], source: "unknown" };
  }
  const colourful = pixels.filter((pixel) => pixel.saturation >= 52);
  const brightNeutral = pixels.filter((pixel) => pixel.saturation <= 45 && pixel.value >= 150);
  const colourRatio = colourful.length / pixels.length;
  const whiteRatio = brightNeutral.length / pixels.length;
  if (whiteRatio >= 0.56 && colourRatio <= 0.24) {
    return {
      quality: "白",
      confidence: Math.min(0.92, 0.55 + whiteRatio * 0.42),
      evidence: `neutral=${whiteRatio.toFixed(2)};saturated=${colourRatio.toFixed(2)}`,
      warnings: [],
      source: "visual_background",
    };
  }
  if (colourRatio < 0.16) {
    return {
      quality: null,
      confidence: 0,
      evidence: `saturated=${colourRatio.toFixed(2)}`,
      warnings: ["quality_low_saturation", "quality_unknown"],
      source: "unknown",
    };
  }
  const hue = median(colourful.map((pixel) => pixel.hue));
  const candidates = Object.entries(QUALITY_HUES).map(([quality, target]) => ({ quality: quality                         , distance: hueDistance(hue, target) }));
  candidates.sort((left, right) => left.distance - right.distance);
  const winner = candidates[0] ;
  if (winner.distance > 19 || colourRatio < 0.34) {
    return {
      quality: null,
      confidence: 0,
      evidence: `hue=${hue.toFixed(1)};distance=${winner.distance.toFixed(1)};saturated=${colourRatio.toFixed(2)}`,
      warnings: ["quality_visual_conflict", "quality_unknown"],
      source: "unknown",
    };
  }
  return {
    quality: winner.quality,
    confidence: Math.min(0.93, 0.40 + colourRatio * 0.38 + (19 - winner.distance) / 19 * 0.25),
    evidence: `hue=${hue.toFixed(1)};distance=${winner.distance.toFixed(1)};saturated=${colourRatio.toFixed(2)}`,
    warnings: [],
    source: "visual_background",
  };
}

export function extractRelativeQualityPixels(image           , rect      )             {
  const left = Math.max(0, Math.floor(rect.x));
  const top = Math.max(0, Math.floor(rect.y));
  const width = Math.max(0, Math.min(image.width - left, Math.ceil(rect.width)));
  const height = Math.max(0, Math.min(image.height - top, Math.ceil(rect.height)));
  if (!width || !height) return [];
  const pixels             = [];
  for (let localY = 0; localY < height; localY += 1) {
    for (let localX = 0; localX < width; localX += 1) {
      const radius = Math.sqrt(
        ((localX - (width - 1) / 2) / (width / 2)) ** 2
        + ((localY - (height - 1) / 2) / (height / 2)) ** 2,
      );
      if (radius < 0.62 || radius > 0.96) continue;
      const at = ((top + localY) * image.width + left + localX) * 4;
      const [hue, saturation, value] = rgbToOpenCvHsv(
        image.data[at] ?? 0,
        image.data[at + 1] ?? 0,
        image.data[at + 2] ?? 0,
      );
      pixels.push({ hue, saturation, value });
    }
  }
  return pixels;
}

export function recognizeQuality(image           , card               )                     {
  if (card.completeness !== "complete") {
    return {
      quality: null,
      confidence: 0,
      evidence: "incomplete_card_quality_not_evaluated",
      warnings: ["quality_incomplete_card", "quality_unknown"],
      source: "unknown",
    };
  }
  return classifyQualityPixels(extractRelativeQualityPixels(image, qualityRectForCard(card)));
}
