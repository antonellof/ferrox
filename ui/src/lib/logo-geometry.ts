/**
 * The Ferrox mark, as numbers.
 *
 * Ferrox is iron oxide, and the structure of α-iron — ferrite — is
 * body-centred cubic. Seen down its body diagonal a cubic cell projects
 * to a regular hexagon with three visible edges meeting at the centre,
 * and the body-centred atom sits exactly on that meeting point. So the
 * mark is a real thing rather than a decoration: an outline (the cell),
 * three spokes (the visible edges) and one filled node (the atom at the
 * centre of the cell). It is drawn in strokes of `currentColor`, so it
 * inherits the text colour in either theme and at any size, and at 16px
 * it collapses to a legible hexagon-with-a-core rather than to mush.
 *
 * The geometry lives here, and not inside the component, because it is
 * drawn TWICE: once by `components/logo.tsx` and once by
 * `public/favicon.svg`, which is a static file no bundler links to the
 * component. Two structures that must agree about one thing is this
 * repo's dominant defect shape, so `logo-geometry.test.ts` reads the
 * favicon off disk and asserts it carries these exact strings.
 */

/** Everything below is expressed in this box. */
export const VIEWBOX = "0 0 24 24";

/**
 * The cell outline: a pointy-top hexagon of radius 9 about (12, 12).
 * Vertices at 90°, 30°, 330°, 270°, 210°, 150°; 9·cos30° = 7.79.
 */
export const CELL_PATH =
  "M12 3 19.79 7.5 19.79 16.5 12 21 4.21 16.5 4.21 7.5Z";

/**
 * The three cube edges that stay visible in this projection, running
 * from the centre out to alternating vertices.
 */
export const SPOKES_PATH = "M12 12 12 3M12 12 19.79 16.5M12 12 4.21 16.5";

/** The body-centred atom. */
export const CORE_R = 2.2;

/** Stroke weight at the 24-unit scale: 1.6 gives ~1.07px at 16px. */
export const STROKE_WIDTH = 1.6;
