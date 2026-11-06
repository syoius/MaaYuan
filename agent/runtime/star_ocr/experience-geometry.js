import { edgeMap, ringScore } from "./main-grid.js";


const COUNT_X_OFFSET = 0.40;
const COUNT_Y_OFFSET = 0.78;
const COUNT_WIDTH = 0.60;
const COUNT_HEIGHT = 0.24;

function median(values          )         {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]  : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

export function experienceCountRect(icon      )       {
  return {
    x: icon.x + Math.trunc(icon.width * COUNT_X_OFFSET),
    y: icon.y + Math.trunc(icon.height * COUNT_Y_OFFSET),
    width: Math.max(1, Math.trunc(icon.width * COUNT_WIDTH)),
    height: Math.max(1, Math.trunc(icon.height * COUNT_HEIGHT)),
  };
}

export function classifyExperienceCompleteness(rect      , bounds      )               {
  if (rect.x < bounds.x || rect.x + rect.width > bounds.x + bounds.width) return "invalid";
  if (rect.y < bounds.y) return "partial_top";
  if (rect.y + rect.height > bounds.y + bounds.height) return "partial_bottom";
  return "complete";
}

export function rgbToOpenCvHsv(red        , green        , blue        )                           {
  const r = red / 255;
  const g = green / 255;
  const b = blue / 255;
  const maximum = Math.max(r, g, b);
  const minimum = Math.min(r, g, b);
  const delta = maximum - minimum;
  let hue = 0;
  if (delta) {
    if (maximum === r) hue = 60 * (((g - b) / delta) % 6);
    else if (maximum === g) hue = 60 * ((b - r) / delta + 2);
    else hue = 60 * ((r - g) / delta + 4);
  }
  if (hue < 0) hue += 360;
  return [hue / 2, maximum ? delta / maximum * 255 : 0, maximum * 255];
}

export function selectedExperienceTabEvidence(image           , viewport      )                                                                {
  const top = viewport.y + Math.trunc(viewport.height * 0.07);
  const bottom = viewport.y + Math.trunc(viewport.height * 0.24);
  const step = 2;
  const width = Math.max(1, Math.ceil(viewport.width / step));
  const height = Math.max(1, Math.ceil((bottom - top) / step));
  const mask = new Uint8Array(width * height);
  for (let localY = 0; localY < height; localY += 1) {
    for (let localX = 0; localX < width; localX += 1) {
      const x = Math.min(image.width - 1, viewport.x + localX * step);
      const y = Math.min(image.height - 1, top + localY * step);
      const at = (y * image.width + x) * 4;
      const [hue, saturation, value] = rgbToOpenCvHsv(image.data[at] ?? 0, image.data[at + 1] ?? 0, image.data[at + 2] ?? 0);
      if (hue >= 8 && hue <= 42 && saturation >= 25 && saturation <= 210 && value >= 120) mask[localY * width + localX] = 1;
    }
  }
  const visited = new Uint8Array(mask.length);
  const choices                                          = [];
  const queue           = [];
  for (let seed = 0; seed < mask.length; seed += 1) {
    if (!mask[seed] || visited[seed]) continue;
    queue.length = 0;
    queue.push(seed);
    visited[seed] = 1;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const point = queue[cursor] ;
      const y = Math.floor(point / width);
      const x = point - y * width;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const next = ny * width + nx;
          if (mask[next] && !visited[next]) { visited[next] = 1; queue.push(next); }
        }
      }
    }
    const componentWidth = (maxX - minX + 1) * step;
    const componentHeight = (maxY - minY + 1) * step;
    const aspect = componentWidth / Math.max(1, componentHeight);
    if (componentWidth >= viewport.width * 0.14 && aspect >= 1.4 && aspect <= 8 && componentHeight <= viewport.height * 0.10) {
      choices.push({ area: componentWidth * componentHeight, center: (minX + maxX + 1) * step / 2, rect: { x: viewport.x + minX * step, y: top + minY * step, width: componentWidth, height: componentHeight } });
    }
  }
  const best = choices.sort((left, right) => right.area - left.area)[0];
  if (!best || best.center < viewport.width * 0.65) return { selected: false, confidence: 0, evidence: [] };
  return { selected: true, confidence: 0.82, rect: best.rect, evidence: ["selected_tab_visual:experience"] };
}

export function experienceSearchBounds(profile, selectedTab) {
  const viewport = profile.viewport;
  // The selected tab is an observed rectangle, not a fixed screenshot y.
  // Keep the single item row below it; currency/header and the capacity badge
  // are outside this region in both portrait profile families.
  const tabBottom = selectedTab?.rect ? selectedTab.rect.y + selectedTab.rect.height : viewport.y + viewport.height * .18;
  const top = Math.max(profile.contentBounds.y, Math.ceil(tabBottom + viewport.width * .025));
  const bottom = Math.min(viewport.y + viewport.height * .75, Math.ceil(top + viewport.width * .32));
  return { x: viewport.x, y: top, width: viewport.width, height: Math.max(0, bottom - top) };
}

function dominantRow(circles, profile)                   {
  const clusters                     = [];
  for (const circle of [...circles].sort((left, right) => left.centerY - right.centerY)) {
    const cluster = clusters.find((item) => Math.abs(circle.centerY - median(item.map((entry) => entry.centerY))) <= Math.max(18, median(item.map((entry) => entry.radius)) * 0.38));
    if (cluster) cluster.push(circle);
    else clusters.push([circle]);
  }
  const selected = clusters.map(row => normalizeExperienceRow(row, profile)).sort((left, right) => right.length - left.length || right.reduce((s,c) => s+c.score,0) - left.reduce((s,c) => s+c.score,0))[0] ?? [];
  const separated                   = [];
  for (const circle of [...selected].sort((left, right) => right.score - left.score)) {
    if (separated.some((item) => Math.abs(item.centerX - circle.centerX) <= Math.min(item.radius, circle.radius) * 1.15)) continue;
    separated.push(circle);
  }
  return separated.sort((left, right) => left.centerX - right.centerX);
}

function normalizeExperienceRow(row                  , profile                   )                   {
  if (!row.length) return [];
  const ratios = profile.profileId === "tablet_portrait_v1" ? [0.15, 0.345, 0.54] : [0.165, 0.38, 0.595];
  const anchors = ratios.map((ratio) => profile.viewport.x + profile.viewport.width * ratio);
  const slots = new Map                        ();
  for (const circle of row) {
    const slot = anchors.reduce((best, anchor, index) => Math.abs(anchor - circle.centerX) < Math.abs(anchors[best]  - circle.centerX) ? index : best, 0);
    if (Math.abs(anchors[slot]  - circle.centerX) > profile.viewport.width * 0.075) continue;
    const previous = slots.get(slot);
    if (!previous || circle.score > previous.score) slots.set(slot, circle);
  }
  const selected = [...slots.entries()].sort(([left], [right]) => left - right);
  const radius = Math.round(median(selected.map(([, circle]) => circle.radius)));
  const centerY = Math.round(median(selected.map(([, circle]) => circle.centerY)));
  return selected.map(([slot, circle]) => ({ ...circle, centerX: Math.round(anchors[slot] ), centerY, radius }));
}

export function findExperienceCircles(image           , profile                   , selectedTab         )                                                                       {
  const bounds = experienceSearchBounds(profile, selectedTab);
  const edges = edgeMap(image);
  const expectedRadius = profile.viewport.width * (profile.profileId === "tablet_portrait_v1" ? 0.061 : 0.068);
  const minRadius = Math.max(18, Math.round(expectedRadius * 0.82));
  const maxRadius = Math.max(minRadius + 2, Math.round(expectedRadius * 1.18));
  const radiusStep = Math.max(3, Math.round(expectedRadius * 0.08));
  const xStep = Math.max(6, Math.round(expectedRadius * 0.14));
  const yStep = Math.max(4, Math.round(expectedRadius * 0.10));
  const raw                   = [];
  for (let centerY = bounds.y + minRadius; centerY <= bounds.y + bounds.height - minRadius; centerY += yStep) {
    for (let centerX = bounds.x + minRadius; centerX <= bounds.x + bounds.width - minRadius; centerX += xStep) {
      let bestRadius = minRadius;
      let bestScore = 0;
      for (let radius = minRadius; radius <= maxRadius; radius += radiusStep) {
        const score = ringScore(edges, image.width, image.height, centerX, centerY, radius);
        if (score > bestScore) { bestScore = score; bestRadius = radius; }
      }
      if (bestScore >= 21) raw.push({ centerX, centerY, radius: bestRadius, score: Math.round(bestScore * 100) / 100, source: "canvas_ring" });
    }
  }
  const proposals                   = [];
  for (const proposal of [...raw].sort((left, right) => right.score - left.score)) {
    if (proposals.some((item) => Math.hypot(item.centerX - proposal.centerX, item.centerY - proposal.centerY) <= Math.min(item.radius, proposal.radius) * 1.15)) continue;
    proposals.push(proposal);
    if (proposals.length >= 48) break;
  }
  const row = dominantRow(proposals, profile);
  return { proposals, row, bounds, coverage: experienceCoverage(image, profile, selectedTab, row, bounds) };
}

export function buildExperienceCandidates(circles                  , bounds      )                        {
  return circles.map((circle, index) => {
    const iconRect = {
      x: Math.trunc(circle.centerX - circle.radius),
      y: Math.trunc(circle.centerY - circle.radius),
      width: Math.max(1, Math.trunc(circle.radius * 2)),
      height: Math.max(1, Math.trunc(circle.radius * 2)),
    };
    const completeness = classifyExperienceCompleteness(iconRect, bounds);
    return {
      itemId: `experience-${index + 1}`,
      index,
      iconRect,
      countRect: experienceCountRect(iconRect),
      circle,
      completeness,
      evidence: completeness === "complete" ? ["experience_icon_complete"] : [completeness],
    };
  });
}

export function experienceCoverage(image, profile, tab, row, bounds) {
  const warnings = [];
  const v = profile.viewport;
  if (!tab?.selected || !tab.rect) warnings.push("experience_tab_bounds_unverified");
  if (!["phone_portrait_v1", "phone_9_16_v1"].includes(profile.profileId)) warnings.push("experience_portrait_profile_unverified");
  if (row.length > 3) warnings.push("experience_too_many_icons");
  const ratios = [.165, .38, .595];
  const slots = row.map(c => ratios.reduce((best, ratio, i) => Math.abs(v.x+ratio*v.width-c.centerX)<Math.abs(v.x+ratios[best]*v.width-c.centerX)?i:best,0));
  if (slots.some((slot,i) => slot!==i)) warnings.push("experience_row_not_left_compacted");
  if (row.some(c => c.centerY-c.radius<bounds.y || c.centerY+c.radius>bounds.y+bounds.height)) warnings.push("experience_icon_outside_content");
  let unexplained = 0;
  // The right quarter contains no stock slot. Its scanline median estimates
  // the backdrop without requiring a saved empty-page template.
  for (let y=bounds.y;y<bounds.y+bounds.height;y+=2) {
    const samples=[[],[],[]];
    for (let x=Math.floor(v.x+v.width*.75);x<v.x+v.width;x+=4) {
      const at=(y*image.width+x)*4;
      for (let channel=0;channel<3;channel++) samples[channel].push(image.data[at+channel]);
    }
    const bg=samples.map(median);
    for (let x=bounds.x;x<bounds.x+bounds.width;x+=2) {
      if (row.some(c => Math.abs(x-c.centerX)<=c.radius*1.2 && Math.abs(y-c.centerY)<=c.radius*1.2)) continue;
      const at=(y*image.width+x)*4;
      const delta=Math.max(...bg.map((value,channel)=>Math.abs(image.data[at+channel]-value)));
      if (delta>45) unexplained+=4;
    }
  }
  const limit=Math.max(80, v.width*v.width*.0015);
  if (unexplained>limit) warnings.push("experience_content_unaccounted");
  const complete=warnings.length===0;
  return { complete, emptyConfirmed: complete && row.length===0, unexplainedPixels: unexplained, unexplainedLimit: limit, warnings,
    evidence: complete ? ["selected_tab_bounds", "complete_compacted_row", "no_unaccounted_foreground"] : [] };
}
