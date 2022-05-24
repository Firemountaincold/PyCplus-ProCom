#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <vector>
#define BuffSize 1024
#define MaxReceive 256 * 256 * 3

using namespace std;

class DataMine
{
private:
    int socketclient;
    sockaddr_in addr;
    int ret;
    int err;

public:
    DataMine(const char *, int);
    int RunFunction(char *, int *, int *, int);
    int CloseSocket();
};

DataMine::DataMine(const char *ip = "127.0.0.1", int port = 8888)
{
    //连接socket
    this->socketclient = socket(AF_INET, SOCK_STREAM, 0);
    this->addr.sin_family = AF_INET;
    this->addr.sin_port = htons(port);
    this->addr.sin_addr.s_addr = inet_addr(ip);
    this->ret = connect(this->socketclient, (sockaddr *)&this->addr, sizeof(sockaddr));
}

int DataMine::RunFunction(char *funcname, int *timestamp, int *data, int length)
{
    // 调用函数
    int l = length * 20 + 20;
    printf("【信息】选择方法：%s\n", funcname);
    char *func = new char[l];
    sprintf(func, "%s,", funcname);
    for (int i = 0; i < length; i++)
    {
        sprintf(func, "%s,%d", func, timestamp[i]);
    }
    sprintf(func, "%s,", func);
    for (int i = 0; i < length; i++)
    {
        sprintf(func, "%s,%d", func, data[i]);
    }
    int siz = send(socketclient, func, l, 0);
    printf("【信息】数据已发送：（%d 字符）%d Bytes\n", siz, siz * 8);

    char re[l]; //收到的字符串

    int ns1 = recv(socketclient, re, l, 0); //收到的字符串的长度
    re[ns1] = '\0';
    printf("【信息】收到回复：（%d 字符）\n%s\n", ns1, re);
    return ns1;
}

int DataMine::CloseSocket()
{
    int ret = shutdown(socketclient, 2);
    return ret;
}

int Run(int *timestamp, int *data, int length, int func)
{
    DataMine dm;
    int ret;
    switch (func)
    {
    case 0:
        ret = dm.RunFunction((char *)"ARIMA", timestamp, data, length); //ARIMA预测
        break;
    case 1:
        ret = dm.RunFunction((char *)"ADTK", timestamp, data, length); //ADTK异常检测
        break;
    case 2:
        ret = dm.RunFunction((char *)"AutoEncoder", timestamp, data, length); //自编码预测
        break;
    case 3:
        ret = dm.RunFunction((char *)"TCN", timestamp, data, length); //自编码预测
        break;
    default:
        break;
    }
    dm.CloseSocket();
    return 0;
}

int standard_to_stamp(const char *str_time)
{
    struct tm stm;
    int iY, iM, iD, iH = 0, iMin = 0, iS = 0;

    memset(&stm, 0, sizeof(stm));
    iY = atoi(str_time + 1);
    iM = atoi(str_time + 6);
    iD = atoi(str_time + 9);

    stm.tm_year = iY - 1900;
    stm.tm_mon = iM - 1;
    stm.tm_mday = iD;
    stm.tm_hour = iH;
    stm.tm_min = iMin;
    stm.tm_sec = iS;

    return (int)mktime(&stm);
}

int readcsv(string csv, int *timestamp, int *data, int length)
{
    ifstream inFile(csv, ios::in);
    if (!inFile)
    {
        cout << "【信息】打开文件失败！" << endl;
        exit(1);
    }
    cout << "【信息】正在读取数据......" << endl;
    int i = 0;
    string line;
    while (i <= length && getline(inFile, line)) // getline(inFile, line)表示按行读取CSV文件中的数据
    {
        string field;
        istringstream sin(line); //将整行字符串line读入到字符串流sin中

        getline(sin, field, ',');                        //将字符串流sin中的字符读入到field字符串中，以逗号为分隔符
        timestamp[i] = standard_to_stamp(field.c_str()); //将刚刚读取的字符串转换成int

        getline(sin, field);           //将字符串流sin中的字符读入到field字符串中
        data[i] = atoi(field.c_str()); //将刚刚读取的字符串转换成int

        i++;
    }
    cout << "【信息】读取完成。" << endl;
    inFile.close();
    return i;
}

int main(int argc, char *argv[])
{
    if(argc==1)
    {
        printf("使用说明：\n参数0：ARIMA方法\n参数1：ADTK方法\n参数2：AutoEncoder方法\n参数3：TCN方法\n");
        return 1;
    }
    string csvf = "temp.csv";
    int length = 3640;
    int *timestamp = new int[length];
    int *data = new int[length];
    readcsv(csvf, timestamp, data, length);
    int func = atoi(argv[1]);
    Run(timestamp, data, length, func);
    return 0;
}