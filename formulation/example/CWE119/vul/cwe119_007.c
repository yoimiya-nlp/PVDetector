#include <stdio.h>

void buffer_overflow3(char *input) {
    char buffer[15];
    sprintf(buffer, "Input: %s", input);
    printf("%s\n", buffer);
}

int main() {
    buffer_overflow3("This is too long");
    return 0;
}