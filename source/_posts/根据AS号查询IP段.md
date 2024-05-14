---
title: 使用Python从AS号查询IP段的方法与实现
date: 2024-05-14 11:38:36
categories: 分享
tags: [分享]
---
在网络管理和安全监控中，了解特定的自治系统（AS）号与其管理的IP地址范围之间的关系是非常重要的。IP地址范围通常用于确定特定网络实体的活动范围，因此能够从给定的AS号查询相关的IP地址段是一项有用的任务。

本文将介绍如何使用Python语言从给定的AS号查询相关的IP地址段，以及实现该过程的方法。

### 1. AS号与IP地址的关系

自治系统号（AS号）是互联网中的一个重要概念，它是分配给互联网服务提供商（ISP）或其他网络实体的唯一标识符。每个AS号都管理着一组IP地址，这些IP地址可以用于路由和网络通信。

### 2. 从AS号查询IP地址段的方法

要从给定的AS号查询相关的IP地址段，可以通过访问WHOIS数据库或使用网络爬虫从特定网站上提取信息。在本文中，我们将介绍如何使用Python编程语言结合网络爬虫技术来实现这一目标。

### 3. 实现过程

首先，我们需要选择一个提供了AS号与IP地址对应关系的网站作为数据源。例如，可以选择像bgp.he.net这样的网站，它提供了从AS号查询IP地址段的功能。

接下来，我们可以编写Python代码来实现以下步骤：

- 使用urllib库发送HTTP请求，从选择的网站上获取包含所需信息的页面内容。
- 使用正则表达式或者BeautifulSoup等库从页面内容中提取出与IP地址相关的信息，例如CIDR地址段。
- 将提取的IP地址段存储在合适的数据结构中，例如列表或者集合。
- 可选：对提取的IP地址段进行排序或者其他处理，以便进一步分析或者使用。

最后，我们可以将代码封装为一个可重用的函数或者类，以便在需要查询IP地址段时直接调用。

### 4. 示例代码

以下是一个简单的示例代码，演示了如何从bgp.he.net查询指定AS号的IP地址段：

```python
#!/usr/bin/python
import urllib.request  # urllib2在Python 3中已被替换为urllib.request
import sys
import os
import re
import string

class AS_TO_ACL:
    def __init__(self, asnum):
        self.asnum = asnum
        self.url = "https://bgp.he.net/AS%s#_prefixes" % (self.asnum)
        self.cidr = set()

    def http_client(self, url):
        # 修改为urllib.request.Request
        request = urllib.request.Request(url, headers={'User-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"})
        try:
            # 修改为urllib.request.urlopen
            response = urllib.request.urlopen(request, timeout=5)
            info = response.info()
            # Python 3中urlopen返回的是bytes而不是str，需要解码为str
            data = response.read().decode('utf-8')
        except urllib.error.HTTPError as error:  # 修改为urllib.error.HTTPError
            print("%s error:%s" % (url, error.reason))
            return None
        except urllib.error.URLError as error:  # 修改为urllib.error.URLError
            print(error.reason)
            return None
        else:
            outdata = data
        return outdata

    def get_acl(self):
        htmldata = self.http_client(self.url)
        if htmldata is None:
            return
        ip_reg = re.compile("/net/(\d+\.\d+\.\d+\.\d+/\d+)")
        # Python 3中split()不再接受参数，默认按空格分割
        htmls = htmldata.split()
        for line in htmls:
            match = ip_reg.search(line)
            if match:
                ips = match.group(1)
                # 使用strip()而不是string.strip()
                self.cidr.add(ips.strip())

        for ip in self.cidr:
            print("%s\n" % (ip), end='')  # Python 3中print()默认结束符为换行符

query = AS_TO_ACL("45102")
query.get_acl()
```

### 5. 结论

通过本文介绍的方法，我们可以轻松地使用Python从给定的AS号查询相关的IP地址段。这为网络管理、安全监控和其他相关领域提供了一种简单而有效的方法来获取所需的网络信息。