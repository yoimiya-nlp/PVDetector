void buffer_overflow_memcpy1(char *input) {
    char buffer[10];
    // 没有检查输入的长度，input过长会导致溢出
    memcpy(buffer, input, strlen(input)-10);
    printf("Buffer: %s\n", buffer);
}

int main() {
    char input[50] = "This is a long string causing overflow";
    buffer_overflow_memcpy1(input);
    return 0;
}