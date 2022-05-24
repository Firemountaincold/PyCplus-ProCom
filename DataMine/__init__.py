import numpy as np
import pandas as pd
import socket
from datetime import datetime
from threading import Thread
import time
import os
from operator import methodcaller


class DataMine:
    def __init__(self):
        self.g_conn_pool = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.thread = Thread(target=self._accept_client)  # 主线程
        self.thread.setDaemon(True)

    def run(self, ip="localhost", port=8888, max_n=0):
        self.server.bind((ip, port))
        self.server.listen(max_n)
        self.thread.start()

        log = """--------------------------\n--------------------------
%s\t\n(ip=%s,port=%d)\n\t开始监听...
""" % (str(datetime.now()), ip, port)
        if os.path.exists('./log.txt'):
            with open('./log.txt', 'a', encoding='utf-8') as fo:
                fo.write(log)
        else:
            with open('./log.txt', 'w', encoding='utf-8') as fo:
                fo.write(log)
        print(log)
        while True:
            cmd = input("""--------------------------
    输入1:查看当前在线人数
    输入0:关闭服务器
--------------------------
""")
            if cmd == '1':
                print("--------------------------")
                print("当前在线人数：\n", len(self.g_conn_pool))
            elif cmd == '0':
                exit()

    def _data_handle(self, client, address):  # 此函数主要用于解析数据，调用函数处理，并将结构序列化为字符串
        func_name, timestamps, datas = self._recvall(client)
        if(func_name==None):
            client.close()
            return
        tag_str, return_data = methodcaller(func_name, timestamps, datas)(self)

        data_str = str(np.array(return_data).reshape(-1).tolist()).replace(' ', '')
        self._sendall(client, tag_str + ":" + data_str)
        time.sleep(2)
        client.close()
        self.g_conn_pool.remove(client)
        log = ('%s\t用户(ip:%s port:%d)\t已下线\n' %
               (str(datetime.now()), address[0], address[1]))
        with open('./log.txt', 'a', encoding='utf-8') as fo:
            fo.write(log)
        print(log)

    def _recvall(self, sock):  # 收到数据，处理数据
        header = sock.recv(6553500)
        header = str(header).split('\\', 1)[0].replace("b'", '')
        try:
            func_name, timestamp, data = header.split(',,')
        except:
            print("【警告】收到格式不规范的数据！长度：{}".format(len(header)))
            return None,None,None
        timestamps = timestamp.split(',')
        datas = data.split(',')
        timestamps = list(map(int, timestamps))
        datas = list(map(int, datas))
        print("【信息】收到数据！\n【信息】调用函数：" + func_name)
        print("【信息】收到时间戳", len(timestamps), "个，大小", len(timestamp)*8, ' Bytes')
        print("【信息】收到数据", len(datas), "个，大小", len(data)*8, ' Bytes')
        return func_name, timestamps, datas
            
    def _sendall(self, sock, data_str):  # 返回数据
        data_bt = bytes(data_str, encoding="ascii")
        total = len(data_str)
        sended = 0
        while sended < total:
            send = min(total-sended, 128*128*3)
            sock.send(data_bt[sended:sended+send])
            sended += send

    def _accept_client(self):  # 此函数用于接收客户端的链接请求，并为其开启一个线程
        while True:
            client, address = self.server.accept()  # 阻塞，等待客户端连接
            # 加入连接池
            self.g_conn_pool.append(client)
            log = ('%s\t用户(ip:%s port:%d)\t已上线\n' %
                   (str(datetime.now()), address[0], address[1]))
            with open('./log.txt', 'a', encoding='utf-8') as fo:
                fo.write(log)
            print(log)
            # 给每个客户端创建一个独立的线程进行管理
            thread = Thread(target=self._data_handle, args=(client, address))
            # 设置成守护线程
            thread.setDaemon(True)
            thread.start()
            
