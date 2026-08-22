/** Zero-pad task IDs to 4 digits so lists line up visually
 *  (e.g. #0007 / #0042 / #1024 instead of #7 / #42 / #1024).
 *  IDs that already exceed 4 digits are returned unpadded. */
export function formatTaskId(id: number | string | null | undefined): string {
  if (id === null || id === undefined) return '#----';
  const s = String(id);
  return '#' + s.padStart(4, '0');
}
