# -*- coding: utf-8 -*-
# @Time    : 2022/1/29 1:30
# @Author  : Justus
# @FileName: Commonly modules.py
# @Software: PyCharm
import time
from datetime import *
from collections import *
import os
import argparse
import base64

# ##########datetime##########
print("\ndatetime:")
# 获取当前日期和时间
print("获取当前日期和时间:")
now = datetime.now()
print(now)
print(type(now))
# timestamp转换为datetime
print("timestamp转换为datetime:")
dt = datetime(2005, 4, 19, 12, 20)
print(dt)
print(dt.timestamp())
# datetime转化为timestamp
print("datetime转化为timestamp:")
t = 1500000000.0
print(datetime.fromtimestamp(t))
print(datetime.utcfromtimestamp(t))
# str转换为datetime
print("str转换为datetime:")
cday = datetime.strptime("2015-6-1 18:19:59", "%Y-%m-%d %H:%M:%S")
print(cday)
# datetime转换为str
print("datetime转换为str:")
print(now.strftime("%a, %b %d %H:%M"))
# datetime加减
print("datetime加减:")
print(now + timedelta(hours=1))
print(now + timedelta(days=1))
print(now + timedelta(days=1, hours=1))
# 转换UTC时间
print("转换UTC时间:")
utc_8 = timezone(timedelta(hours=8))
print(now)
print(now.replace(tzinfo=utc_8))
# 时区转换
print("时区转换:")
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
print(utc_dt)
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
print(bj_dt)
tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt)
tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt2)

# ##########collections##########
print("\ncollections:")
# namedtuple
print("namedtuple:")
Point = namedtuple("Point", ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)
print("deque:")
q = deque(['a', 'b', 'c'])
print(q)
q.append('x')
print(q)
q.appendleft('y')
print(q)
q.pop()
print(q)
q.popleft()
print(q)
# OrderedDict
print("OrderedDict:")
d = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
print(d)
d['z'] = 4
d['y'] = 5
d['x'] = 6
print(d)


# OrderedDict实现先进先出
class LastUpdatedOrderedDict(OrderedDict):
    def __init__(self, capacity):
        super(LastUpdatedOrderedDict, self).__init__()
        self._capacity = capacity

    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print('remove:', last)
        if containsKey:
            del self[key]
            print('set:', (key, value))
        else:
            print('add:', (key, value))
        OrderedDict.__setitem__(self, key, value)


# ##########ChainMap##########
print("\nChainMap:")
defaults = {
    'color': 'red',
    'user': 'guest'
}
# 构造命令行参数:
parser = argparse.ArgumentParser()
print(parser.parse_args())
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
print(parser.parse_args())
namespace = parser.parse_args()
command_line_args = { k: v for k, v in vars(namespace).items() if v }
# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)
# 打印参数:
print('color=%s' % combined['color'])
print('user=%s' % combined['user'])

# ##########Counter##########
print("\nCounter:")
c = Counter()
c.update("programming")
print(c)
c.update("running")
print(c)

# ##########Base64##########
print("\nBase64:")
print(base64.b64encode(b'binary\x00string'))
print(base64.b64decode(b'YmluYXJ5AHN0cmluZw=='))
print(base64.b64encode(b'i\xb7\x1d\xfb\xef\xff'))
print(base64.urlsafe_b64encode(b'i\xb7\x1d\xfb\xef\xff'))
print(base64.urlsafe_b64decode('abcd--__'))
