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
    if (len < 99) {
      int i;
      char *str;
      char corrupt[110];
      for (i=0;i<len;i++) {
        if (line[i] < 32 || line[i] > 126)
          line[i] = 95;
      }
      strncpy(corrupt, line, 100);
      /*for (i=0;i<len;i++) {
        if (rand()%50==0) corrupt[i] = 32 + rand()%95;
      }*/
      /*
      str = strstr(corrupt, "their");
      if (!str) str = strstr(corrupt, "there");
      if (str) {
        if (rand()&1) {
          str[3] = 'i';
          str[4] = 'r';
        } else {
          str[3] = 'r';
          str[4] = 'e';
        }
        printf("%s\t%s\n", corrupt, line);
      }*/
#if 0
      int pos = rand()%len-1;
      memmove(corrupt+pos, corrupt+pos+1, len-pos);
#endif
#if 1
      int pos = rand()%len-1;
      if (pos < 0) pos = 0;
      memmove(corrupt+pos+1, corrupt+pos, len-pos);
      line[len] = ' ';
      line[len+1] = 0;
#endif
      printf("%s\t%s\n", corrupt, line);
    }
  }
  return 0;
}
