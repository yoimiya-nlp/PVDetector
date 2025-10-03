int save_input()
{
    int TEMP_NUM = 40;
    int mem_num = 20;
    char temp[TEMP_NUM];
    mem = (char *)malloc(mem_num);
    gets(temp);
    if (strlen(temp) >= TEMP_NUM) {
        printf("Error!input too long!");
    }
    else {
        strcpy(mem, temp);
     }
     return 0;
}