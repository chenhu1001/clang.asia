---
title: ROS自定义Msg消息
date: 2024-04-23 11:02:15
categories: ROS
tags: [ROS]
---
# 自定义消息流程
在Ros中，如果没有现成的消息类型来描述要去传递的消息时，我们会自定义消息。

通常我们会新建一个Package来去自定义消息，这个Package一般不去写任何的业务逻辑，只是用来声明自定义的消息类型，可以只定义一种消息类型，也可以定义多种消息类型，根据业务需求来定。

所以，首先我们单独的创建一个package，我们取名为demo_msgs，一定要要添加roscpp，rospy，rosmsg的依赖。
## 1 . 创建msg目录
在pakage目录下新建msg目录

## 2. 新建msg文件
创建的这个Student.msg文件就是自定义消息文件，需要去描述消息的格式。

我们可以编辑代码如下
```
string name
int64 age
```
这个自定义的消息包含两个数据形式，name和age，name的类型 是string，age的类型是int64。

这个msg文件其实遵循一定规范的，每一行表示一种数据。前面是类型，后面是名称。

ros不只是提供了int64和string两种基本类型供我们描述，其实还有很多，具体可以自行搜索
## 3. 配置package.xml文件
在package.xml种添加如下配置:
```
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```
## 4. 配置CMakeLists.txt
find_package配置

在find_package添加message_generation，结果如下:
```
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  message_generation
)
```
add_message_file配置

添加add_message_file，结果如下:
```
add_message_files(
        FILES
        Student.msg
)
```
> 这里的Student.msg要和你创建的msg文件名称一致，且必须时在msg目录下，否则编译会出现问题  

generation_msg配置
添加generation_msg，结果如下:
```
generate_messages(
        DEPENDENCIES
        std_msgs
)
```
> 这个配置的作用是添加生成消息的依赖，默认的时候要添加std_msgs  

catkin_package配置
修改catkin_package，结果如下:
```
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demo_msg
   CATKIN_DEPENDS roscpp rosmsg rospy message_runtime
#  DEPENDS system_lib
)
```
> 为catkin编译提供了依赖message_runtime
# 检验自定义消息
## 1. 编译项目
来到工作空间目录下，运行编译
```
catkin_make
```
## 2. 查看生成的消息文件
c++的头文件
来到devel的include目录下，如果生成了头文件说明，自定义消息创建成功。

python的py文件
来到devel的lib/python2.7/dist-package目录下，查看是否生成和package名称相同的目录，以及目录内是否生成对应的py文件。

## 3. 通过rosmsg工具校验
```
rosmsg show demo_msgs/Student
```

# 常见问题
## 1. 如果出现msg找不到的情况
清除编译信息并重新编译，

## 2. python导入方式
```
import rospy
from demo_msgs.msg import Student
```
