---
title: "【Go编程基础】08map"
date: 2022-04-29T19:12:06+08:00
categories: ["Golang"]
tags: [Golang]
---
## map
- 类似其它语言中的哈希表或者字典，以key-value形式存储数据
- Key必须是支持==或!=比较运算的类型，不可以是函数、map或slice
- Map查找比线性搜索快很多，但比使用索引访问数据的类型慢100倍
_ Map使用make()创建，支持 := 这种简写方式

- make([keyType]valueType, cap)，cap表示容量，可省略
- 超出容量时会自动扩容，但尽量提供一个合理的初始值
- 使用len()获取元素个数

- 键值对不存在时自动添加，使用delete()删除某键值对
- 使用 for range 对map和slice进行迭代操作

## 思考问题
- 根据在 for range 部分讲解的知识，尝试将类型为map[int]string
的键和值进行交换，变成类型map[string]int
- 程序正确运行后应输出如下结果：
