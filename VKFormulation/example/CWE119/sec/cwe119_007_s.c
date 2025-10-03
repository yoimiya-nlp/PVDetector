#include <stdio.h>

void buffer_overflow3(char *input) {
    char buffer[15];
    printf("%s\n", input);
}

int main() {
    buffer_overflow3("This is too long");
    return 0;
}