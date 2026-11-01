import { candidateFromCircle } from "./card-completeness.js";
import { layoutSpec } from "./profiles.js";


function median(values          )         {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]  : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

export function edgeMap(image           )             {
  const gray = new Uint8Array(image.width * image.height);
  for (let index = 0; index < gray.length; index += 1) {
    const at = index * 4;
    gray[index] = Math.round(((image.data[at] ?? 0) * 299 + (image.data[at + 1] ?? 0) * 587 + (image.data[at + 2] ?? 0) * 114) / 1000);
  }
  const edges = new Uint8Array(gray.length);
  for (let y = 1; y < image.height - 1; y += 1) {
    for (let x = 1; x < image.width - 1; x += 1) {
      const at = y * image.width + x;
      const gx = Math.abs((gray[at + 1] ?? 0) - (gray[at - 1] ?? 0));
      const gy = Math.abs((gray[at + image.width] ?? 0) - (gray[at - image.width] ?? 0));
      edges[at] = Math.min(255, gx + gy);
    }
  }
  return edges;
}

export function ringScore(edges            , width        , height        , x        , y        , radius        )         {
  let score = 0;
  let count = 0;
  for (let index = 0; index < 40; index += 1) {
    const angle = index / 40 * Math.PI * 2;
    for (const offset of [-2, 0, 2]) {
      const px = Math.round(x + Math.cos(angle) * (radius + offset));
      const py = Math.round(y + Math.sin(angle) * (radius + offset));
      if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) continue;
      score += edges[py * width + px] ?? 0;
      count += 1;
    }
  }
  return count ? score / count : 0;
}

function suppressColumn(peaks                  )                   {
  const selected                   = [];
  for (const proposal of [...peaks].sort((a, b) => b.score - a.score)) {
    if (selected.some((item) => Math.abs(item.centerY - proposal.centerY) <= Math.min(item.radius, proposal.radius) * 1.15)) continue;
    selected.push(proposal);
  }
  return selected.sort((a, b) => a.centerY - b.centerY);
}

export function findCircleProposals(image           , profile                   )                   {
  const spec = layoutSpec(profile.profileId);
  const edges = edgeMap(image);
  const expectedRadius = profile.viewport.width * spec.radiusRatio;
  const minRadius = Math.max(12, Math.round(expectedRadius * 0.82));
  const maxRadius = Math.max(minRadius + 2, Math.round(expectedRadius * 1.18));
  const radiusStep = Math.max(3, Math.round(expectedRadius * 0.055));
  const xSearch = Math.round(profile.viewport.width * spec.horizontalSearchRatio);
  const xStep = Math.max(4, Math.round(expectedRadius * 0.12));
  const yStep = Math.max(2, Math.round(profile.viewport.height / 900));
  const proposals                   = [];
  for (const normalizedX of spec.columnCenters) {
    const expectedX = Math.round(profile.viewport.x + profile.viewport.width * normalizedX);
    const columnPeaks                   = [];
    for (let centerY = profile.contentBounds.y - maxRadius; centerY <= profile.viewport.y + profile.viewport.height - minRadius / 2; centerY += yStep) {
      let bestRadius = minRadius;
      let bestScore = 0;
      let bestX = expectedX;
      for (let radius = minRadius; radius <= maxRadius; radius += radiusStep) {
        for (let centerX = expectedX - xSearch; centerX <= expectedX + xSearch; centerX += xStep) {
          const score = ringScore(edges, image.width, image.height, centerX, centerY, radius);
          if (score > bestScore) { bestScore = score; bestRadius = radius; bestX = centerX; }
        }
      }
      if (bestScore >= 21) columnPeaks.push({ centerX: bestX, centerY, radius: bestRadius, score: Math.round(bestScore * 100) / 100, source: "canvas_ring" });
    }
    proposals.push(...suppressColumn(columnPeaks));
  }
  if (!proposals.length) return [];
  const radiusMode = median(proposals.map((item) => item.radius));
  return proposals.filter((item) => Math.abs(item.radius - radiusMode) <= Math.max(4, radiusMode * 0.22));
}



export function groupGridRows(proposals                  , profile                   )            {
  if (!proposals.length) return [];
  const spec = layoutSpec(profile.profileId);
  const columns = spec.columnCenters.map((value) => profile.viewport.x + profile.viewport.width * value);
  const radius = median(proposals.map((item) => item.radius));
  const tolerance = Math.max(12, radius * 0.7);
  const rows            = [];
  for (const proposal of [...proposals].sort((a, b) => a.centerY - b.centerY || b.score - a.score)) {
    const column = columns.reduce((best, current, index) => Math.abs(current - proposal.centerX) < Math.abs(columns[best]  - proposal.centerX) ? index : best, 0);
    if (Math.abs(columns[column]  - proposal.centerX) > profile.viewport.width * 0.11) continue;
    let row = rows.find((item) => Math.abs(item.y - proposal.centerY) <= tolerance);
    if (!row) { row = { y: proposal.centerY, circles: new Map() }; rows.push(row); }
    const previous = row.circles.get(column);
    if (!previous || proposal.score > previous.score) row.circles.set(column, proposal);
    row.y = median([...row.circles.values()].map((item) => item.centerY));
  }
  const rowScore = (row         )         => [...row.circles.values()].reduce((sum, item) => sum + item.score, 0) / Math.max(1, row.circles.size);
  const expectedRowRadius = profile.viewport.width * spec.radiusRatio;
  const radiusEligible = rows.filter((row) => Math.abs(median([...row.circles.values()].map((item) => item.radius)) - expectedRowRadius) <= expectedRowRadius * 0.15);
  const populated = (radiusEligible.filter((row) => row.circles.size >= 2).length >= 2 ? radiusEligible : rows).filter((row) => row.circles.size >= 2);
  const bestRowScore = Math.max(...populated.map(rowScore), 0);
  const evidenceThreshold = bestRowScore * 0.62;
  const evidenceRows = populated.filter((row) => rowScore(row) >= evidenceThreshold);
  const strong = (evidenceRows.length >= 2 ? evidenceRows : populated).sort((a, b) => a.y - b.y);
  if (!strong.length) return [];
  const fullAnchors = strong.filter((row) => row.circles.size === 4);
  const cadenceSource = fullAnchors.length >= 3 ? fullAnchors : strong;
  const gaps = cadenceSource.slice(1).map((row, index) => row.y - cadenceSource[index] .y).filter((gap) => gap >= profile.viewport.height * spec.rowSpacingRange[0] && gap <= profile.viewport.height * spec.rowSpacingRange[1]);
  const cadence = gaps.length ? median(gaps) : 0;
  let accepted = [...strong];
  if (cadence && fullAnchors.length >= 3) {
    const alignmentRows = evidenceRows.length >= 2 ? radiusEligible.filter((row) => rowScore(row) >= evidenceThreshold) : rows;
    const base = [...fullAnchors].sort((left, right) => {
      const alignedCount = (candidate         )         => alignmentRows.filter((row) => Math.abs(row.y - (candidate.y + Math.round((row.y - candidate.y) / cadence) * cadence)) <= cadence * 0.18).reduce((sum, row) => sum + row.circles.size, 0);
      return alignedCount(right) - alignedCount(left) || rowScore(right) - rowScore(left);
    })[0] .y;
    const aligned = new Map                 ();
    for (const row of alignmentRows) {
      const offset = Math.round((row.y - base) / cadence);
      if (Math.abs(row.y - (base + offset * cadence)) > cadence * 0.18) continue;
      const previous = aligned.get(offset);
      const score = [...row.circles.values()].reduce((sum, item) => sum + item.score, 0);
      const previousScore = previous ? [...previous.circles.values()].reduce((sum, item) => sum + item.score, 0) : -1;
      if (!previous || row.circles.size > previous.circles.size || (row.circles.size === previous.circles.size && score > previousScore)) aligned.set(offset, row);
    }
    accepted = [...aligned.entries()].filter(([offset, row]) => row.circles.size >= 2 || offset < 0 || offset > fullAnchors.length - 1).map(([, row]) => row);
  }
  if (cadence) {
    for (const row of rows.filter((item) => item.circles.size === 1 && rowScore(item) >= evidenceThreshold)) {
      const last = accepted.at(-1);
      const first = accepted[0];
      if ((last && row.y > last.y && Math.abs(row.y - last.y - cadence) <= cadence * 0.35)
        || (first && row.y < first.y && Math.abs(first.y - row.y - cadence) <= cadence * 0.35)) accepted.push(row);
    }
  }
  return accepted.sort((a, b) => a.y - b.y);
}

export function buildCardCandidates(proposals                  , profile                   )                  {
  const rows = groupGridRows(proposals, profile);
  const cards                  = [];
  rows.forEach((row, rowIndex) => {
    [...row.circles.entries()].sort(([left], [right]) => left - right).forEach(([columnIndex, circle]) => {
      cards.push(candidateFromCircle(circle, profile, rowIndex, columnIndex, cards.length + 1));
    });
  });
  return cards;
}
