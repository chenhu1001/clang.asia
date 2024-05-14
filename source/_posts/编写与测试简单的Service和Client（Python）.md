---
title: 编写与测试简单的Service和Client（Python）
date: 2024-05-14 14:39:15
categories: ROS
tags: [ROS]
---
# 01 导读

**C++ 代码必须通过编译生成可执行文件；**

**python 代码是可执行文件，不需要编译；**

* 开发的功能包都放在 catkin_ws 这样一个工作空间里；
* 新建的功能包取名为 service_example，实现两个整数求和为例，client 端节点向 server 端节点发送 a、b 的请求，server 端节点返回响应 sum=a+b 给 client 端节点；

**服务编程流程**

* 创建服务器
* 创建客户端
* 添加编译选项
* 运行可执行程序

# 02 功能包的创建

在 catkin_ws/src/ 目录下新建功能包 service_example，并在创建时显式的指明依赖 rospy 和 std_msgs，依赖 std_msgs 将作为基本数据类型用于定义我们的服务类型。打开命令行终端，输入命令：

```java
$ cd ~/catkin_ws/src

# 创建功能包 topic_example 时，显式的指明依赖 rospy 和 std_msgs，
# 依赖会被默认写到功能包的 CMakeLists.txt 和 package.xml 中
$ catkin_create_pkg service_example rospy std_msgs
```

# 03 在功能包中创建自定义服务类型

* 服务(srv): 一个 srv 文件描述一项服务。它包含两个部分：请求和响应。

*服务类型的定义文件都是以*.srv 为扩展名，srv 文件则存放在功能包的 srv 目录下。

* 服务通信过程中服务的数据类型需要用户自己定义，与消息不同，节点并不提供标准服务类型。

## 3.1 定义 srv 文件

srv 文件分为请求和响应两部分，由 '---' 分隔。

在功能包 service_example 目录下新建 srv 目录，然后在 service_example/srv/ 目录中创建 AddTwoInts.srv 文件

```java
int64 a
int64 b
---
int64 sum
```

其中`a`和`b`是请求，而`sum` 是响应。

## 3.2 在 package.xml 中添加功能包依赖

srv 文件被转换成为 C++，Python 和其他语言的源代码：

查看 `package.xml`, 确保它包含一下两条语句:

```java
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

如果没有，添加进去。 注意，在构建的时候，我们只需要"message_generation"。然而，在运行的时候，我们只需要"message_runtime"。

## 3.3 在 CMakeLists.txt 添加编译选项

**第一步，增加 message_generation**

打开功能包中的 CMakeLists.txt 文件，利用 find_packag 函数，增加对 `message_generation` 的依赖，这样就可以生成消息了。 你可以直接在 `COMPONENTS` 的列表里增加 `message_generation`，就像这样：

```java
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  )
```

有时候你会发现，即使你没有调用 find_package, 你也可以编译通过。这是因为 catkin 把你所有的功能包都整合在一起，因此，如果其他的功能包调用了 find_package，你的功能包的依赖就会是同样的配置。但是，在你单独编译时，忘记调用 find_package 会很容易出错。

**第二步，删掉 `#`，去除对下边语句的注释:**

找到如下代码块:

```java
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )
```

用你自己定义的 srv 文件名（AddTwoInts.srv）替换掉那些`Service*.srv`文件，修改好后的代码如下：

```java
add_service_files(
  FILES
  AddTwoInts.srv
)
```

**第三步，msg 和 srv 都需要的步骤**

在 `CMakeLists.txt` 中找到如下部分:

```java
# generate_messages(
#   DEPENDENCIES
# #  std_msgs  # Or other packages containing msgs
# )
```

去掉注释并附加上所有你消息文件所依赖的那些含有`.msg`文件的功能包（这个例子是依赖`std_msgs`, 不要添加 roscpp,rospy)，结果如下:

```java
generate_messages(
  DEPENDENCIES
  std_msgs
)
```

原因：generate_messages 的作用是自动创建我们自定义的消息类型 .msg 与服务类型 .srv 相对应的 .h，由于我们定义的服务类型使用了 std_msgs 中的 int64 基本类型，所以必须向 generate_messages 指明该依赖。

**第四步，由于增加了新的消息，所以我们需要重新编译我们的功能包：**

目的：查看配置是否有问题

```java
$ cd ~/catkin_ws
$ catkin_make -DCATKIN_WHITELIST_PACKAGES="service_example"
```

所有在 msg 路径下的.msg 文件都将转换为 ROS 所支持语言的源代码。生成的 C++ 头文件将会放置在`~/catkin_ws/devel/include/service_example/`。 Python 脚本语言会在`~/catkin_ws/devel/lib/python2.7/dist-packages/service_example/msg` 目录下创建。

# 04 查看自定义的服务消息

通过 <功能包名 / 服务类型名> 找到该服务，打开命令行终端，输入命令：

```java
$ source ~/catkin_ws/devel/setup.bash

$ rossrv show service_example/AddTwoInts
```

# 05 功能包的源代码编写

功能包中需要编写两个独立可执行的节点，一个节点用来作为 client 端发起请求，另一个节点用来作为 server 端响应请求，所以需要在新建的功能包 service_example/scripts 目录下新建两个文件 server.py 和 client.py，并将下面的代码分别填入。

## 5.1 编写 Service 节点（server.py）

将创建一个简单的 service 节点("server")，该节点将接收到两个整形数字，并返回它们的和。

**如何实现一个服务器**

* 初始化 ROS 节点；
* 创建 Server 实例；
* 循环等待服务请求，进入回调函数；
* 在回调函数中完成服务功能的处理，并反馈应答数据。

在 service_example 包中创建 scripts /server.py 文件：

```python
#!/usr/bin/env python

from service_example.srv import AddTwoInts,AddTwoIntsResponse
import rospy

def handle_add_two_ints(req):
    print ("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
 
 #因为我们已经将服务的类型声明为AddTwoInts，所以它会为您生成AddTwoIntsRequest对象（可以自由传递）
    return AddTwoIntsResponse(req.a + req.b)    # AddTwoIntsResponse由服务生成的返回函数

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')  # 声明节点为add_two_ints_server
   
    #定义服务器节点名称，服务类型，处理函数
        #处理函数调用实例化的AddTwoIntsRequest接收请求和返回实例化的AddTwoIntsResponse
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print ("Ready to add two ints.")
    rospy.spin()   # 就像订阅者示例一样，rospy.spin()使代码不会退出，直到服务关闭；

if __name__ == "__main__":
    add_two_ints_server()
```

在～/catkin_ws/src/service_example 下，让节点可执行:

```java
$ chmod +x scripts/server.py
```

## 5.2 编写 Client 节点（client.py）

**如何实现一个客户端**

* 初始化 ROS 节点；
* 创建一个 Client 实例；
* 发布服务请求数据；
* 等待 Server 处理之后的应答结果。

在 service_example 包中创建 scripts /client.py 文件，并在其中粘贴以下內容：

```python
#!/usr/bin/env python

""" 
导入sys模块，sys.argv的功能是在外部向程序的内部传递参数。sys.argv(number)，number=0的时候是脚本的名称
"""
import sys
import rospy
from service_example.srv import *

def add_two_ints_client(x, y):
    # 等待服务节点的接入
    rospy.wait_for_service('add_two_ints')
    try: 
        # 创建服务的处理句柄，可以像调用函数一样调用句柄
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        # 创建服务请求对象
        request = AddTwoIntsRequest()
        request.a = x
        request.b = y
        # 调用服务并获取响应
        resp = add_two_ints(request)
        # 返回响应中的和
        return resp.sum 
    except rospy.ServiceException as e:
        # 如果调用失败，可能会抛出rospy.ServiceException
        print("Service call failed: %s" % e)

def usage():
    return "%s [x y]" % sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print(usage())
        sys.exit(1)
    print("Requesting %s + %s" % (x, y))
    print("%s + %s = %s" % (x, y, add_two_ints_client(x, y)))
```

在～/catkin_ws/src/service_example 节点可执行：

```java
$ chmod +x scripts/client.py
```

**代码解析：**

我们可以像普通函数一样使用这个句柄并调用它：

```java
resp1 = add_two_ints(x, y)
return resp1.sum
```

因为我们已经将服务的类型声明为 AddTwoInts，所以它会为你生成 AddTwoIntsRequest 对象(可以自由传递)。返回值是 AddTwoIntsResponse 对象。如果调用失败，可能会抛出 rospy.ServiceException，因此你应该设置适当的 try/except 块。

# 06 功能包的编译

我们使用 CMake 作为构建系统，是的，即使是 Python 节点也必须使用它。这是为了确保创建消息和服务时自动生成 Python 代码。

```java
$ cd ~/catkin_ws
$ catkin_make -DCATKIN_WHITELIST_PACKAGES="service_example"
```

# 07 测试 service 和 client

## 7.1 运行 Service

第一步，打开一个命令行终端：

```java
$ roscore
```

第二步，打开第二个命令行终端：

```java
# 用 rosrun <package_name> <node_name> 启动功能包中的发布节点。
$ source ~/catkin_ws/devel/setup.bash    # 激活 catkin_ws 工作空间（必须有，必不可少）
$ rosrun service_example server.py       # (python 版本)
```

你将看到如下的输出信息:

```java
Ready to add two ints.              # Server 节点启动后的日志信息
```

## 7.2 运行 Client

现在，运行 Client 并附带一些参数：

打开第三个命令行客户端：

```java
$ source ~/catkin_ws/devel/setup.bash     # 激活 catkin_ws 工作空间（必须有，必不可少）
$ rosrun service_example client.py 1 3    # (Python)
```

你将会看到如下的输出信息:

```java
# Client 启动后发布服务请求，并成功接收到反馈结果
Requesting 1+3
1 + 3 = 4

# Server 接收到服务调用后完成加法求解，并将结果反馈给 Client
Returning [1 + 3 = 4]
```

现在，你已经成功地运行了你的第一个 Service 和 Client 程序。