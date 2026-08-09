/* Diagnostic validator built against PurismCore, which logs the exact field
 * that failed consistency -- the official Core only returns 0/1. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Live2DCubismCore.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); if(!f){perror("open");return 2;}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  void*buf=NULL; posix_memalign(&buf,64,n+64); memset(buf,0,n+64);
  fread(buf,1,n,f); fclose(f);
  printf("== %s (%ld bytes) ==\n", argv[1], n);
  printf("version=%u\n", csmGetMocVersion(buf,(unsigned)n));
  int c=csmHasMocConsistency(buf,(unsigned)n);
  printf("consistency=%d\n", c);
  if(!c) return 1;
  csmMoc*moc=csmReviveMocInPlace(buf,(unsigned)n);
  if(!moc){printf("revive failed\n");return 1;}
  unsigned sz=csmGetSizeofModel(moc); void*mb=NULL; posix_memalign(&mb,256,sz); memset(mb,0,sz);
  csmModel*mdl=csmInitializeModelInPlace(moc,mb,sz);
  printf("model=%p drawables=%d\n",(void*)mdl, mdl?csmGetDrawableCount(mdl):-1);
  return 0;
}
