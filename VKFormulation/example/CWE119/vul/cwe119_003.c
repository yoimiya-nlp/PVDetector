int save_input()
{
    int TEMP_NUM = 40;
    int mem_num = 80;
    char temp[TEMP_NUM];
    mem_num = mem_num - 50;
    mem = (char *)malloc(mem_num);
    gets(temp);

    strcpy(mem, temp);
    return 0;
}