#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main()
{
  char *line=NULL;
  size_t alloc_size=0;
  while (1) {
    int len = getline(&line, &alloc_size, stdin);
    if (len == -1) break;
    line[len-1] = 0;
    len--;
    //printf ("%d %d %d %d\n", len, alloc_size, (int)line[len], (int)line[len-1]);
    if (len < 100) {
      int i;
      char corrupt[110];
      for (i=0;i<len;i++) {
        if (line[i] < 32 || line[i] > 126)
          line[i] = 95;
      }
      strncpy(corrupt, line, 100);
      for (i=0;i<len;i++) {
        if (rand()%50==0) corrupt[i] = 32 + rand()%95;
      }
      printf("%s\t%s\n", corrupt, line);
    }
  }
  return 0;
}
