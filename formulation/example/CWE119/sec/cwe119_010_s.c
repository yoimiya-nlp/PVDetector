void buffer_overflow_memcpy2(char *input, size_t length) {
    char buffer[20];
    memcpy(buffer, input, 10);
    printf("Buffer: %s\n", buffer);
}

int main() {
    char input[50] = "Another long string causing overflow";
    buffer_overflow_memcpy2(input, 40);
    return 0;
}