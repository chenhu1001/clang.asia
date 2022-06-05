---
title: "【Go编程基础】08map"
date: 2022-06-05T21:31:36+08:00
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

```go
func main() {
	m := make(map[int]string)
	m[1]="OK"
	a := m[1]
	fmt.Println(a)
	fmt.Println(m)
}

输出：
OK
map[1:OK]
```

```go
func main() {
	var m map[int]map[int]string
	m = make(map[int]map[int]string)
	m[1] = make(map[int]string)
	m[1][1] = "OK"
	a := m[1][1]
	fmt.Println(a)

	// 多返回值用于判断map的value是否存在
	a, ok := m[2][1]
	fmt.Println(a, ok)
	if !ok {
		m[2] = make(map[int]string)
	}
	m[2][1] = "GOOD"
	a, ok = m[2][1]
	fmt.Println(a, ok)
}

输出：
OK
 false
GOOD true
```

```go
func main() {
	sm := make([]map[int]string, 5)
	for _, v := range sm {
		v = make(map[int]string, 1)
		v[1] = "OK"
		fmt.Println(v)
	}
	fmt.Println(sm)
}

输出：
map[1:OK]
map[1:OK]
map[1:OK]
map[1:OK]
map[1:OK]
[map[] map[] map[] map[] map[]]
```

```go
func main() {
	sm := make([]map[int]string, 5)
	for i := range sm {
		sm[i] = make(map[int]string, 1)
		sm[i][1] = "OK"
		fmt.Println(sm[i])
	}
	fmt.Println(sm)
}

输出：
map[1:OK]
map[1:OK]
map[1:OK]
map[1:OK]
map[1:OK]
[map[1:OK] map[1:OK] map[1:OK] map[1:OK] map[1:OK]]
```

## 思考问题
- 根据在 for range 部分讲解的知识，尝试将类型为map[int]string
的键和值进行交换，变成类型map[string]int
- 程序正确运行后应输出如下结果：
