#include <stdio.h>

void buffer_overflow2() {
    char buffer[20];
    printf("Enter some text: ");
    gets(buffer); 
    printf("You entered: %s\n", buffer);
}

int main() {
    buffer_overflow2();
    return 0;
}