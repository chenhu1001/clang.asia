---
title: "【Go编程基础】13-反射reflection"
date: 2022-04-28T14:32:30+08:00
categories: ["Golang"]
tags: [Golang]
---
# 反射reflection
- 反射可大大提高程序的灵活性，使得 interface{} 有更大的发挥余地
- 反射使用 TypeOf 和 ValueOf 函数从接口中获取目标对象信息
- 反射会将匿名字段作为独立字段（匿名字段本质）
- 想要利用反射修改对象状态，前提是 interface.data 是 settable，
即 pointer-interface
- 通过反射可以“动态”调用方法

```go
type User struct {
	Id int
	Name string
	Age int
}

func (u User) Hello() {
	fmt.Println("Hello world.")
}

func main() {
	u := User{1, "OK", 18}
	Info(u)
}

func Info(o interface{}) {
	t := reflect.TypeOf(o)
	fmt.Println("Type:", t.Name())
	
	// 反射结构的判断
	if k := t.Kind(); k != reflect.Struct {
		fmt.Println("XX")
		return
	}

	v := reflect.ValueOf(o)
	fmt.Println("Fields")

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		val := v.Field(i).Interface()
		fmt.Printf("%6s: %v = %v\n", f.Name, f.Type, val)
	}
	
	for i := 0; i < t.NumMethod(); i++ {
		m := t.Method(i)
		fmt.Printf("%6s: %v\n", m.Name, m.Type)
	}
}

输出：
Type: User
Fields
    Id: int = 1
  Name: string = OK
   Age: int = 18
 Hello: func(main.User)
```

# 思考问题
- 定义一个结构，通过反射来打印其信息，并调用方法。
