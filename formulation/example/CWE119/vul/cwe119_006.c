#include <stdio.h>

void buffer_overflow4(char *input) {
    char buffer[12];
    for (int i = 0; i < 20; i++) {
        buffer[i] = input[i];
    }
    printf("Buffer: %s\n", buffer);
}

int main() {
    buffer_overflow4("Overflow happens here!");
    return 0;
}