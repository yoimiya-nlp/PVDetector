#include <stdio.h>

void buffer_overflow5() {
    char buffer[8];
    for (int i = 0; i <= 5; i++) { 
        buffer[i] = 'A';
    }
    printf("Buffer: %s\n", buffer);
}

int main() {
    buffer_overflow5();
    return 0;
}