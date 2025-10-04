#include <stdio.h>
#include <string.h>

void buffer_overflow1(char *input) {
    char buffer[10];
    strcpy(buffer, input);  
    printf("Buffer: %s\n", buffer);
}

int main() {
    char input[50] = "This is a long string causing overflow";
    buffer_overflow1(input);
    return 0;
}