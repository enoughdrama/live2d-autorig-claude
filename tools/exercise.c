/* Drive every parameter through its range and report which drawables move.
 *
 * A rig that loads but does not deform is the most common silent failure --
 * consistency checks pass, the model appears, and nothing animates. This is
 * the check that catches it. Reports per-parameter how many drawables respond
 * and the largest vertex displacement, so a parameter wired to nothing is
 * obvious.
 *
 * Usage: exercise <file.moc3>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Live2DCubismCore.h"

static void *slurp(const char *path, long *n_out)
{
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  void *buf = NULL;
  if (posix_memalign(&buf, 64, n + 64)) { fclose(f); return NULL; }
  memset(buf, 0, n + 64);
  if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
  fclose(f);
  *n_out = n;
  return buf;
}

int main(int argc, char **argv)
{
  if (argc < 2) { fprintf(stderr, "usage: %s <file.moc3>\n", argv[0]); return 2; }
  long n = 0;
  void *buf = slurp(argv[1], &n);
  if (!buf) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }

  if (!csmHasMocConsistency(buf, (unsigned)n)) { printf("REJECTED\n"); return 1; }
  csmMoc *moc = csmReviveMocInPlace(buf, (unsigned)n);
  unsigned msz = csmGetSizeofModel(moc);
  void *mb = NULL;
  if (posix_memalign(&mb, 256, msz)) return 2;
  memset(mb, 0, msz);
  csmModel *mdl = csmInitializeModelInPlace(moc, mb, msz);
  csmUpdateModel(mdl);

  int dc = csmGetDrawableCount(mdl);
  int pc = csmGetParameterCount(mdl);
  const int *vc = csmGetDrawableVertexCounts(mdl);
  const csmVector2 **vp = csmGetDrawableVertexPositions(mdl);
  const char **pids = csmGetParameterIds(mdl);
  const float *pmin = csmGetParameterMinimumValues(mdl);
  const float *pmax = csmGetParameterMaximumValues(mdl);
  const float *pdef = csmGetParameterDefaultValues(mdl);
  float *pv = csmGetParameterValues(mdl);

  /* snapshot the rest pose */
  int total = 0;
  for (int i = 0; i < dc; i++) total += vc[i];
  csmVector2 *rest = malloc(sizeof(csmVector2) * total);
  int *base = malloc(sizeof(int) * dc);
  int k = 0;
  for (int i = 0; i < dc; i++) { base[i] = k; for (int v = 0; v < vc[i]; v++) rest[k++] = vp[i][v]; }

  printf("%-20s %-8s %-8s %s\n", "parameter", "moved", "maxdelta", "value");
  int dead = 0;
  for (int p = 0; p < pc; p++) {
    for (int q = 0; q < pc; q++) pv[q] = pdef[q];
    /* drive to whichever extreme is further from the default */
    float lo = pmin[p], hi = pmax[p], d = pdef[p];
    float target = (fabsf(hi - d) >= fabsf(d - lo)) ? hi : lo;
    pv[p] = target;
    csmUpdateModel(mdl);

    int moved = 0; double worst = 0.0;
    for (int i = 0; i < dc; i++) {
      int any = 0;
      for (int v = 0; v < vc[i]; v++) {
        double dx = vp[i][v].X - rest[base[i] + v].X;
        double dy = vp[i][v].Y - rest[base[i] + v].Y;
        double m = sqrt(dx * dx + dy * dy);
        if (m > 1e-6) any = 1;
        if (m > worst) worst = m;
      }
      moved += any;
    }
    if (!moved) dead++;
    printf("%-20s %-8d %-8.4f %.2f%s\n", pids[p], moved, worst, target,
           moved ? "" : "   <-- DRIVES NOTHING");
  }

  printf("\n%d/%d parameters drive at least one drawable\n", pc - dead, pc);
  return dead ? 1 : 0;
}
