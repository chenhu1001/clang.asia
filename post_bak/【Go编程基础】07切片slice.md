---
title: "【Go编程基础】07切片slice"
date: 2022-05-28T18:31:58+08:00
categories: ["Golang"]
tags: [Golang]
---
切片Slice

其本身并不是数组，它指向底层的数组
作为变长数组的替代方案，可以关联底层数组的局部或全部
为引用类型
可以直接创建或从底层数组获取生成
使用len()获取元素个数，cap()获取容量
一般使用make()创建
如果多个slice指向相同底层数组，其中一个的值改变会影响全部

make([]T, len, cap)
其中cap可以省略，则和len的值相同
len表示存数的元素个数，cap表示容量

Slice与底层数组的对应关系
Reslice

Reslice时索引以被slice的切片为准
索引不可以超过被slice的切片的容量cap()值
索引越界不会导致底层数组的重新分配而是引发错误

Append
可以在slice尾部追加元素
可以将一个slice追加在另一个slice尾部
如果最终长度未超过追加到slice的容量则返回原始slice
如果超过追加到的slice的容量则将重新分配数组并拷贝原始数据

Copy

