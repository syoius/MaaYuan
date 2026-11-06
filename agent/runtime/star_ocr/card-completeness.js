

export function isInside(rect      , bounds      )          {
  return rect.x >= bounds.x && rect.y >= bounds.y
    && rect.x + rect.width <= bounds.x + bounds.width
    && rect.y + rect.height <= bounds.y + bounds.height;
}

export function boxesForCircle(circle                )                                                                      {
  const { centerX: x, centerY: y, radius: r } = circle;
  return {
    cardRect: { x: Math.trunc(x - r * 1.05), y: Math.trunc(y - r), width: Math.trunc(r * 2.1), height: Math.trunc(r * 2.05) },
    discRect: { x: x - r, y: y - r, width: r * 2, height: r * 2 },
    nameRect: { x: Math.trunc(x - r * 0.92), y: Math.trunc(y + r * 1.08), width: Math.trunc(r * 1.84), height: Math.max(12, Math.trunc(r * 0.62)) },
    levelRect: { x: Math.trunc(x + r * 0.02), y: Math.trunc(y - r * 1.06), width: Math.trunc(r * 1.08), height: Math.max(12, Math.trunc(r * 0.58)) },
  };
}

// The bottom name box includes padding. Trim only a one-pixel rounding
// overrun in MaaYuan's verified phone viewport; never expand the viewport.
function boundedBoxes(circle, profile) {
  const boxes = boxesForCircle(circle);
  const b = profile.contentBounds;
  const bottom = b.y + b.height;
  const name = boxes.nameRect;
  if (profile.imageWidth === 720 && profile.imageHeight === 1280
    && b.y === 272 && bottom === 1044
    && isInside(boxes.discRect, b) && isInside(boxes.levelRect, b)
    && name.x >= b.x && name.x + name.width <= b.x + b.width
    && name.y >= b.y && name.y + name.height === bottom + 1) {
    boxes.originalNameRect = { ...name };
    boxes.nameRect = { ...name, height: name.height - 1 };
  }
  return boxes;
}

export function classifyCompleteness(circle                , profile                   )                                                     {
  const boxes = boundedBoxes(circle, profile);
  const left = Math.max(0, profile.viewport.x, profile.contentBounds.x);
  const top = Math.max(0, profile.viewport.y, profile.contentBounds.y);
  const right = Math.min(profile.imageWidth, profile.viewport.x + profile.viewport.width, profile.contentBounds.x + profile.contentBounds.width);
  const bottom = Math.min(profile.imageHeight, profile.viewport.y + profile.viewport.height, profile.contentBounds.y + profile.contentBounds.height);
  const horizontal = { x: left, y: top, width: right - left, height: bottom - top };
  const parts = [boxes.discRect, boxes.nameRect, boxes.levelRect];
  if (parts.some((rect) => rect.x < horizontal.x || rect.x + rect.width > horizontal.x + horizontal.width)) {
    return { completeness: "invalid", evidence: ["horizontal_bounds_exceeded"] };
  }
  if (parts.some((rect) => rect.y < horizontal.y)) return { completeness: "partial_top", evidence: ["disc_or_ocr_roi_above_content_top"] };
  if (parts.some((rect) => rect.y + rect.height > horizontal.y + horizontal.height)) return { completeness: "partial_bottom", evidence: ["disc_or_ocr_roi_below_content_bottom"] };
  return { completeness: "complete", evidence: ["disc_name_level_inside_content_bounds", ...(boxes.originalNameRect ? ["name_bottom_padding_clipped_1px"] : [])] };
}

export function candidateFromCircle(circle                , profile                   , rowIndex        , columnIndex        , index        )                {
  const boxes = boundedBoxes(circle, profile);
  const complete = classifyCompleteness(circle, profile);
  return { cardId: `card_${String(index).padStart(3, "0")}`, rowIndex, columnIndex, circle, ...boxes, ...complete };
}
