/* moc3 validation oracle, built against the official Cubism Core.
 *
 * csmHasMocConsistency is the ground truth: it walks every section table and
 * rejects a file the runtime would choke on. A file that passes here and
 * initializes a model is structurally valid -- whether it *looks* right is a
 * separate question only a human can answer.
 *
 * Build: see tools/build_oracle.sh
 * Usage: validate <file.moc3> [param_index param_value]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Live2DCubismCore.h"

int main(int argc, char **argv)
{
  if (argc < 2) { fprintf(stderr, "usage: %s <file.moc3> [param_idx value]\n", argv[0]); return 2; }

  FILE *f = fopen(argv[1], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);

  void *buf = NULL;
  if (posix_memalign(&buf, 64, n + 64)) { fprintf(stderr, "oom\n"); return 2; }
  memset(buf, 0, n + 64);
  if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
  fclose(f);

  printf("%-28s size=%7ld ", argv[1], n);

  unsigned int ver = csmGetMocVersion(buf, (unsigned)n);
  int consistent = csmHasMocConsistency(buf, (unsigned)n);
  printf("mocVersion=%u consistency=%d\n", ver, consistent);
  if (!consistent) { printf(" -> REJECTED by csmHasMocConsistency\n"); return 1; }

  csmMoc *moc = csmReviveMocInPlace(buf, (unsigned)n);
  if (!moc) { printf(" -> csmReviveMocInPlace failed\n"); return 1; }

  unsigned msz = csmGetSizeofModel(moc);
  void *mb = NULL;
  if (posix_memalign(&mb, 256, msz)) { fprintf(stderr, "oom\n"); return 2; }
  memset(mb, 0, msz);
  csmModel *mdl = csmInitializeModelInPlace(moc, mb, msz);
  if (!mdl) { printf(" -> csmInitializeModelInPlace failed\n"); return 1; }

  csmUpdateModel(mdl);

  int dc = csmGetDrawableCount(mdl);
  int pc = csmGetParameterCount(mdl);
  int partc = csmGetPartCount(mdl);
  printf(" -> LOADED drawables=%d params=%d parts=%d\n", dc, pc, partc);

  const char **pids = csmGetParameterIds(mdl);
  const float *pmin = csmGetParameterMinimumValues(mdl);
  const float *pmax = csmGetParameterMaximumValues(mdl);
  for (int i = 0; i < pc && i < 12; i++)
    printf("    param[%d] %-24s [%.2f .. %.2f]\n", i, pids[i], pmin[i], pmax[i]);
  if (pc > 12) printf("    ... %d more params\n", pc - 12);

  const int *vc = csmGetDrawableVertexCounts(mdl);
  const int *ic = csmGetDrawableIndexCounts(mdl);
  const char **dids = csmGetDrawableIds(mdl);
  for (int i = 0; i < dc && i < 8; i++)
    printf("    mesh[%d] %-24s verts=%-5d idx=%d\n", i, dids[i], vc[i], ic[i]);
  if (dc > 8) printf("    ... %d more meshes\n", dc - 8);

  /* Optional: drive a parameter and report whether geometry actually moved.
     A rig that loads but never deforms is the most common silent failure. */
  if (argc >= 4) {
    int pi = atoi(argv[2]);
    float pv = (float)atof(argv[3]);
    const csmVector2 **vp = csmGetDrawableVertexPositions(mdl);
    if (pi < 0 || pi >= pc) { printf(" -> bad param index\n"); return 1; }
    csmVector2 before = vp[0][0];
    float *vals = csmGetParameterValues(mdl);
    vals[pi] = pv;
    csmUpdateModel(mdl);
    csmVector2 after = vp[0][0];
    float d = (after.X - before.X) * (after.X - before.X) +
              (after.Y - before.Y) * (after.Y - before.Y);
    printf(" -> %s=%.2f moved v0 by %.6f  (%.4f,%.4f) -> (%.4f,%.4f)\n",
           pids[pi], pv, d > 0 ? 1.0 : 0.0, before.X, before.Y, after.X, after.Y);
    if (d == 0.0f) printf("    WARNING: geometry did not move\n");
  }

  return 0;
}
