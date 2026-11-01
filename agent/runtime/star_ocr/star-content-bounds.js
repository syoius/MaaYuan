

/** Validated only against the controlled 720x1280 MuMu CaptureBatch viewport.
 * Across 101 real images, readable complete level ROIs start at 272 or later;
 * the missed top clips start at 260/266, below the selected-tab bottom (257).
 * Complete required ROIs end at <=1037, clipped bottom ROIs at >=1046.
 * Dimensions validate eligibility, never provenance.
 */
export const MAAYUAN_MUMU_LAYOUT = {
  imageWidth: 720, imageHeight: 1280,
  contentTopRatio: 272 / 1280, contentBottomRatio: 1044 / 1280,
}         ;

export function refineStarContentBounds(
  profile                   , selectedTab             , layoutHint                 ,
)                    {
  const { viewport } = profile;
  let top = profile.contentBounds.y;
  let bottom = top + profile.contentBounds.height;
  const evidence           = [];
  if (selectedTab && selectedTab.width >= viewport.width * .14 && selectedTab.height > 0
    && selectedTab.x >= viewport.x && selectedTab.x + selectedTab.width <= viewport.x + viewport.width
    && selectedTab.y >= viewport.y + viewport.height * .07
    && selectedTab.y + selectedTab.height <= viewport.y + viewport.height * .24) {
    top = Math.max(top, selectedTab.y + selectedTab.height);
    evidence.push(`content_top:selected_tab_bottom:${top}`);
  }
  const layoutMatches = profile.profileId === "phone_9_16_v1"
    && profile.imageWidth === MAAYUAN_MUMU_LAYOUT.imageWidth && profile.imageHeight === MAAYUAN_MUMU_LAYOUT.imageHeight
    && viewport.x === 0 && viewport.y === 0
    && viewport.width === profile.imageWidth && viewport.height === profile.imageHeight;
  if (layoutHint === "maayuan_mumu" && layoutMatches) {
    top = Math.max(top, viewport.y + Math.round(viewport.height * MAAYUAN_MUMU_LAYOUT.contentTopRatio));
    bottom = Math.min(bottom, viewport.y + Math.round(viewport.height * MAAYUAN_MUMU_LAYOUT.contentBottomRatio));
    evidence.push(`content_top:maayuan_mumu_layout:${MAAYUAN_MUMU_LAYOUT.contentTopRatio}`);
    evidence.push(`content_bottom:maayuan_mumu_layout:${MAAYUAN_MUMU_LAYOUT.contentBottomRatio}`);
  }
  if (bottom <= top) return profile;
  return { ...profile, contentBounds: { ...profile.contentBounds, y: top, height: bottom - top }, evidence: [...profile.evidence, ...evidence] };
}
