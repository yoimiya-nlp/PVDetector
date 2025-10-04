#include <stdio.h>

void buffer_overflow2() {
    char buffer[20];
    printf("Enter some text: ");
    fgets(buffer, sizeof(buffer), stdin);
    buffer[strcspn(buffer, "\n")] = 0;
    printf("You entered: %s\n", buffer);
}

int main() {
    buffer_overflow2();
    return 0;
}