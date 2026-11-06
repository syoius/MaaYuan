# Saved tail regression fixtures

Only adjacent-page inventory viewports (x=40..679, y=272..1043) and fields used by position tests are retained. These are game pixels, not complete screenshots or complete scan results; no account or connection data is included.

- `star_full_tail`: real main-003/main-004 failure pair, four entirely new rows with a 30px grid phase change.
- `star_overlap_tail`: real main and support overlapping tail pairs from the earlier accepted scans. Main touch=669px; support touch=656px.

Tests restore crops into a blank 720x1280 canvas. Earlier inventory prefixes in the overlap tests are explicitly synthetic copies with unit IDs/fields; only the final predecessor and tail are real page records. The original tail counts, ordering, field-conflict, atomic-failure and no-move assertions are retained. Pixels across the inventory viewport are required to reject incorrect offset candidates.
