---
title: "【Go编程基础】09函数function"
date: 2022-06-06T21:50:32+08:00
categories: ["Golang"]
tags: [Golang]
---
## 函数function
- Go 函数 不支持 嵌套、重载和默认参数
- 但支持以下特性：  
  - 无需声明原型、不定长度变参、多返回值、命名返回值参数  
  - 匿名函数、闭包
- 定义函数使用关键字 func，且左大括号不能另起一行
- 函数也可以作为一种类型使用

## defer
- 执行方式类似其它语言中的析构函数，在函数体执行结束后按照调用顺序的相反顺序逐个执行
- 即使函数发生严重错误也会执行
- 支持匿名函数的调用
- 常用于资源清理、文件关闭、解锁以及记录时间等操作
- 通过与匿名函数配合可在return之后修改函数计算结果
- 如果函数体内某个变量作为defer时匿名函数的参数，则在定义defer时即已经获得了拷贝，否则则是引用某个变量的地址
- Go 没有异常机制，但有 panic/recover 模式来处理错误
- Panic 可以在任何地方引发，但recover只有在defer调用的函数中有效

## 问题思考
运行以下程序并分析输出结果。

```go
func main() {
	var fs = [4]func(){}

	for i := 0; i < 4; i++ {
		defer fmt.Println("defer i = ", i)
		defer func() { fmt.Println("defer_closure i = ", i) }()
		fs[i] = func() {
			fmt.Println("closure i = ", i)
		}
	}

	for _, f := range fs {
		f()
	}
}
```
