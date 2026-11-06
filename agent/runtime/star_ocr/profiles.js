














export const PHONE_PROFILE                    = {
  profileId: "phone_portrait_v1", deviceKind: "phone", ratioMin: 0.39, ratioMax: 0.50,
  columnCenters: [0.165, 0.380, 0.597, 0.812], gridRegion: [0.05, 0.18, 0.90, 0.94],
  rowSpacingRange: [0.075, 0.135], bottomSafeY: 0.89, radiusRatio: 0.088, horizontalSearchRatio: 0,
};
export const PHONE_9_16_PROFILE                    = {
  // Observed only in the 9:16 phone UI samples. Keep this narrow so normal tablets
  // remain on their established geometry family.
  profileId: "phone_9_16_v1", deviceKind: "phone", ratioMin: 0.50, ratioMax: 0.57,
  columnCenters: [0.165, 0.380, 0.597, 0.812], gridRegion: [0.05, 0.18, 0.90, 0.94],
  rowSpacingRange: [0.075, 0.135], bottomSafeY: 0.89, radiusRatio: 0.079, horizontalSearchRatio: 0,
};
export const TABLET_PROFILE                    = {
  profileId: "tablet_portrait_v1", deviceKind: "tablet", ratioMin: 0.57, ratioMax: 0.78,
  columnCenters: [0.165, 0.380, 0.597, 0.812], gridRegion: [0.05, 0.18, 0.90, 0.94],
  rowSpacingRange: [0.075, 0.145], bottomSafeY: 0.89, radiusRatio: 0.068, horizontalSearchRatio: 0.09,
};
export const FALLBACK_PROFILE                    = {
  profileId: "unknown_portrait_fallback", deviceKind: "unknown", ratioMin: 0.25, ratioMax: 0.90,
  columnCenters: [0.165, 0.380, 0.597, 0.812], gridRegion: [0.05, 0.18, 0.90, 0.94],
  rowSpacingRange: [0.070, 0.150], bottomSafeY: 0.87, radiusRatio: 0.078, horizontalSearchRatio: 0.09,
};

export function layoutSpec(profileId                                )                    {
  return profileId === PHONE_PROFILE.profileId ? PHONE_PROFILE
    : profileId === PHONE_9_16_PROFILE.profileId ? PHONE_9_16_PROFILE
    : profileId === TABLET_PROFILE.profileId ? TABLET_PROFILE : FALLBACK_PROFILE;
}

function edgeDepth(data                   , width        , height        , axis           , reverse         )         {
  const size = axis === "x" ? width : height;
  const limit = Math.max(1, Math.floor(size * 0.18));
  let depth = 0;
  for (let offset = 0; offset < limit; offset += 1) {
    const position = reverse ? size - 1 - offset : offset;
    let sum = 0;
    let squared = 0;
    const count = axis === "x" ? height : width;
    for (let other = 0; other < count; other += 1) {
      const x = axis === "x" ? position : other;
      const y = axis === "y" ? position : other;
      const at = (y * width + x) * 4;
      const gray = ((data[at] ?? 0) * 299 + (data[at + 1] ?? 0) * 587 + (data[at + 2] ?? 0) * 114) / 1000;
      sum += gray;
      squared += gray * gray;
    }
    const mean = sum / count;
    const deviation = Math.sqrt(Math.max(0, squared / count - mean * mean));
    if (mean <= 10 && deviation <= 8) depth += 1;
    else break;
  }
  return depth >= 8 ? depth : 0;
}

export function detectViewport(image           )                                                             {
  const left = edgeDepth(image.data, image.width, image.height, "x", false);
  const right = edgeDepth(image.data, image.width, image.height, "x", true);
  const top = edgeDepth(image.data, image.width, image.height, "y", false);
  const bottom = edgeDepth(image.data, image.width, image.height, "y", true);
  const width = image.width - left - right;
  const height = image.height - top - bottom;
  if (width < image.width * 0.45 || height < image.height * 0.45) {
    return { viewport: { x: 0, y: 0, width: image.width, height: image.height }, confidence: 0.6, evidence: ["black_bar_detection_rejected"] };
  }
  const cropped = left + right + top + bottom > 0;
  return {
    viewport: { x: left, y: top, width, height },
    confidence: cropped ? 0.8 : 0.95,
    evidence: [cropped ? `black_bars:${left},${top},${right},${bottom}` : "black_bars:none"],
  };
}

export function createScreenshotProfile(image           )                    {
  const detected = detectViewport(image);
  const ratio = detected.viewport.width / Math.max(1, detected.viewport.height);
  const spec = ratio >= PHONE_PROFILE.ratioMin && ratio < PHONE_PROFILE.ratioMax ? PHONE_PROFILE
    : ratio >= PHONE_9_16_PROFILE.ratioMin && ratio < PHONE_9_16_PROFILE.ratioMax ? PHONE_9_16_PROFILE
    : ratio >= TABLET_PROFILE.ratioMin && ratio < TABLET_PROFILE.ratioMax ? TABLET_PROFILE : FALLBACK_PROFILE;
  const contentTop = detected.viewport.y + Math.round(detected.viewport.height * spec.gridRegion[1]);
  const contentBottom = detected.viewport.y + Math.round(detected.viewport.height * spec.bottomSafeY);
  return {
    profileId: spec.profileId,
    deviceKind: spec.deviceKind,
    imageWidth: image.width,
    imageHeight: image.height,
    viewport: detected.viewport,
    contentBounds: { x: detected.viewport.x, y: contentTop, width: detected.viewport.width, height: Math.max(0, contentBottom - contentTop) },
    columnCount: 4,
    confidence: detected.confidence,
    evidence: [...detected.evidence, `profile:${spec.profileId}`, `content_top:profile_grid_${spec.gridRegion[1]}`, `content_bottom:profile_safe_${spec.bottomSafeY}`],
  };
}
