---
title: "【Go编程基础】14-并发concurrency"
date: 2022-04-28T14:32:34+08:00
categories: ["Go"]
tags: [Go]
---
# 并发concurrency
- 很多人都是冲着 Go 大肆宣扬的高并发而忍不住跃跃欲试，但其实从
源码的解析来看，goroutine 只是由官方实现的超级“线程池”而已。
不过话说回来，每个实例 4-5KB 的栈内存占用和由于实现机制而大幅
减少的创建和销毁开销，是制造 Go 号称的高并发的根本原因。另外，
goroutine 的简单易用，也在语言层面上给予了开发者巨大的便利。
- 并发不是并行：Concurrency Is Not Parallelism，并发主要由切换时间片来实现“同时”运行，在并行则是直接利用
多核实现多线程的运行，但 Go 可以设置使用核数，以发挥多核计算机
的能力。
- Goroutine 奉行通过通信来共享内存，而不是共享内存来通信。

# Channel
- Channel 是 goroutine 沟通的桥梁，大都是阻塞同步的
- 通过 make 创建，close 关闭
- Channel 是引用类型
- 可以使用 for range 来迭代不断操作 channel
- 可以设置单向或双向通道
- 可以设置缓存大小，在未被填满前不会发生阻塞

# Select
- 可处理一个或多个 channel 的发送与接收
- 同时有多个可用的 channel时按随机顺序处理
- 可用空的 select 来阻塞 main 函数
- 可设置超时

# 思考问题
- 创建一个 goroutine，与主线程按顺序相互发送信息若干次并打印
