# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 12:06
# @Author  : Justus
# @FileName: Entry-Level.py
# @Software: PyCharm
import logging
import math
import os
import types
from collections.abc import *
from functools import *
import functools
import time
import sys
from enum import *
from io import *
import pickle
import json
import re


# ##########函数##########
# 重写abs函数
def re_abs(x):
    # 若x非int和float型抛出异常
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x


print("\n-9的绝对值:", abs(-9))
print("-9的绝对值:", re_abs(-9))


# 定义函数，接收三个参数返回一元二次方程ax^2+bx+c=0的两个解
def quadratic(a, b, c):
    # 若a,b,c非int和float型抛出异常
    if not isinstance(a or b or c, (int, float)):
        raise TypeError('bad operand type')
    x1 = (-b + math.sqrt(b*b - 4*a*c))/(2*a)
    x2 = (-b - math.sqrt(b*b - 4*a*c))/(2*a)
    return x1, x2


print("x^2+3x+2=0的根为:", quadratic(1, 3, 2))


# ##########参数##########
print("\n参数:")


# 默认参数
def power(x, n=2):
    result = 1
    for i1 in range(n):
        result *= x
    return result


print("3的平方:", power(3))
print("3的立方:", power(3, 3))


# 可变参数
def calc_sum(*numbers):
    result = 0
    for i2 in numbers:
        result += i2
    return result


print("1,5,6,9的和为:", calc_sum(1, 5, 6, 9))
nums = [1, 5, 6, 9]
# 转换为可变参数
print("1,5,6,9的和为:", calc_sum(*nums))


# 关键字参数
def person(name, city, **other):
    print('name:', name, 'city:', city, 'other:', other)


person('Justus', 'beijing', gender='Male', job='Engineer')
extra = {'gender': 'Male', 'job': 'Engineer'}
# 字典读取传入
person('Justus', 'beijing', gender=extra['gender'], job=extra['job'])
# 字典转为关键字参数
person('Justus', 'beijing', **extra)


# 命名关键字参数
def person1(name, city, *, gender, job):
    print(name, city, gender, job)


# 无默认值关键字参数必传
person1('Justus', 'beijing', gender='Male', job='Engineer')


def person2(name, city, *, gender='Male', job='Engineer'):
    print(name, city, gender, job)


# 有默认值可不传参数
person2('Justus', 'beijing')
# 修改默认值
person2('Justus', 'beijing', gender='Female', job='None')


# 组合使用
# 必选参数 > 默认参数 > 可变参数 > 命名关键字参数 > 关键字参数
def combination1(a, b, c=0, *args_, **kw_):
    print('a=', a, 'b=', b, 'c=', c, 'args=', args_, 'kw=', kw_)


def combination2(a, b, c=0, *, d_, **kw_):
    print('a=', a, 'b=', b, 'c=', c, 'd=', d_, 'kw=', kw_)


combination1(1, 2)
combination1(1, 2, c=3)
combination1(1, 2, 3, 'a', 'b')
combination1(1, 2, 3, 'a', 'b', x=99)
combination2(1, 2, d_=99, ext=None)
# 任意函数都可通过function(*args, **kw)的形式调用
args = (1, 2, 3, 4)
kw = {'d': 99, 'x': '#'}
combination1(*args, **kw)

args = (1, 2, 3)
kw = {'d_': 88, 'x': '#'}
combination2(*args, **kw)


# ##########递归##########
def factorial(n):
    if n == 1:
        return 1
    else:
        return n*factorial(n-1)


print("\n5的阶乘(普通递归):", factorial(5))


# 尾递归优化
def fact(n):
    return fact_iter(n, 1)


def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num-1, num * product)


print("5的阶乘(尾递归优化):", fact(5))


# 汉诺塔
def hanoi(n, a, b, c):
    if n == 1:
        print(a, "——>", c)
    else:
        hanoi(n-1, a, c, b)  # 将上层N-1个块由A经C至B
        hanoi(1, a, b, c)  # 将最大的块由A至C
        hanoi(n-1, b, a, c)  # 将N-1个块由B经A至C


print("\n3层汉诺塔步骤:")
hanoi(3, 'A', 'B', 'C')

# ##########切片##########
print("\n切片:")
list_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("lis:\t\t",  list_)
print("[0:3]:\t\t", list_[0:3])
print("[:3]:\t\t", list_[:3])
print("[-2:]:\t\t", list_[-2:])
print("[1:5:2]:\t", list_[1:5:2])
print("[::2]:\t\t", list_[::2])


# 去除字符串的首尾空格
def trim(s):
    if not isinstance(s, str):
        raise TypeError('bad operand type')
    head = 0
    tail = 0
    # 统计开头空格，遇到非空格break
    for i3 in range(len(s)):
        if s[i3] == " ":
            head += 1
        elif s[i3] != " ":
            break
    # 统计结尾空格，遇到非空格break
    for i4 in range(len(s)):
        if s[len(s)-i4-1] == " ":
            tail += 1
        elif s[len(s)-i4-1] != " ":
            break
    return s[head:-tail]


print("\n去掉字符串前后空格:")
str_ = " 这是 一个 字符串 "
print('@', str_, '@')
print('@', str_.strip(" "), '@')
print('@', trim(str_), '@')

# ##########迭代##########
print("\n迭代字典:")
d = {'a': 1, 'b': 2, 'c': 3}
for key_, value in d.items():
    print(key_, value)

# 判断是否可迭代
print("\n判断是否可迭代:")
print("字符串:", isinstance('123', Iterable))
print("整数:", isinstance(123, Iterable))
print("列表:", isinstance([1, 2, 3], Iterable))
print("元组:", isinstance((1, 2, 3), Iterable))
print("字典:", isinstance({'a': 1, 'b': 2, 'c': 3}, Iterable))


# 用迭代找list中的最大值最小值
def find_mm(l_):
    if not l_:
        return None, None
    else:
        max_ = l_[0]
        min_ = l_[0]
        for i5 in l_:
            if i5 > max_:
                max_ = i5
            if i5 < min_:
                min_ = i5
    return min_, max_


print("\n通过迭代找出[7, 1, 3, 9, 5]最大最小值:")
print(find_mm([7, 1, 3, 9, 5]))

# ##########列表生成式##########
print("\n列表生成式:")
print([x for x in range(11)])
print([x * x for x in range(11)])
print([x * x for x in range(11) if x % 2 == 0])
print([m + n for m in "ABC" for n in "XYZ"])
print([-x for x in range(1, 11)])
print([x if x % 2 == 0 else -x for x in range(1, 11)])

# ##########生成器##########
print("\n生成器:")
g = (x for x in range(5))
print(g)
print(next(g))
print(next(g))
print(next(g))
g = (x for x in range(5))
for i in g:
    print(i)


# 菲波那切数列
def ord_fib(n):
    a, b, i_ = 1, 1, 0
    while i_ < n:
        print("ord_fib:", a)
        a, b = b, a+b
        i_ += 1
    return 'done'


print("\n传统斐波那契:")
print(ord_fib(6))


def gen_fib(n):
    a, b, i_ = 1, 1, 0
    while i_ < n:
        yield a
        a, b = b, a + b
        i_ += 1
    return 'done'


print("\n生成器斐波那契:")
for i in gen_fib(6):
    print("gen_fib:", i)
# 得到返回值
g = gen_fib(6)
while 1:
    try:
        x = next(g)
    except StopIteration as e:
        print("Generator return value:", e.value)
        break


# 杨辉三角
def pascal(n):
    if not isinstance(n, int):
        raise TypeError('bad operand type')
    if n == 0:
        return None
    if n == 1:
        yield [1]
    if n == 2:
        yield [1, 1]
    else:
        s = [1]
        yield s
        s = [1, 1]
        yield s
        # 从1到n-2层
        for i6 in range(1, n-1):
            # 初始化temp为[1,1],插入指针k为0
            temp = [1, 1]
            k = 0
            # j是当前层要插的数量
            for j in range(i6):
                temp.insert(j+1, s[k]+s[k+1])
                k += 1
            s = temp
            yield s


print("\n杨辉三角:")
for i7 in pascal(7):
    print(i7)


# 杨辉三角 答案
def triangles():
    l1 = [1]
    while 1:
        yield l1
        x = [0] + l1
        y = l1 + [0]
        l1 = [x[i8] + y[i8] for i8 in range(len(x))]


print("\n杨辉三角(答案):")
g = triangles()
for i10 in range(7):
    print(next(g))


# ##########迭代器##########
# 用iter()函数将可迭代对象变为迭代器
print("\n转换为迭代器:")
print("str是否为迭代器:", isinstance('', Iterator))
print("转换后:", isinstance(iter(''), Iterator))
print("list是否为迭代器:", isinstance([], Iterator))
print("转换后:", isinstance(iter([]), Iterator))
print("dict是否为迭代器:", isinstance({}, Iterator))
print("转换后:", isinstance(iter({}), Iterator))


# ##########map和reduce##########
def add(a, b, fun):
    return fun(a) + fun(b)


print("|-3|+|-4|=", add(-3, -4, abs))


# 元素平方
def f(x):
    return x * x


s1 = [1, 2, 3, 4, 5]
print("\n将", s1, "元素平方:")
r = map(f, s1)
print(r)
print(list(r))


# 元素拼接
def fn(a, b):
    return a * 10 + b


print("\n将", s1, "元素拼接:")
print(reduce(fn, s1))


# 规范人名大小写
def norm_name(na):
    return na.capitalize()


name1 = ["AlEn", "bOb", "cArRY"]
print("\n规范人名[\"AlEn\", \"bOb\", \"cArRY\"]大小写:")
print(list(map(norm_name, name1)))


# 元素求积
def prod(a1, b1):
    return a1 * b1


print("\n将", s1, "元素求积:")
print(reduce(prod, s1))


# 字符串转浮点数
def integer(a, b):
    return float(a) * 10 + float(b)


def decimal(a, b):
    return float(a) * 0.1 + float(b)


def s2f(s2):
    int_ = reduce(integer, s2.split('.')[0])
    dec_ = reduce(decimal, s2.split('.')[1][::-1])
    return int_ + 0.1 * dec_


s3 = "123.456"
print("\n将", s3, "str元素转为float:")
print(s2f(s3))


# 字符串转浮点数(答案)
def str2float(s4_):
    i11 = list(map(lambda x: DIGITS[x], s4_))
    i12 = len(i11)
    for i12 in range(len(i11)):
        if i11[i12] == '.':
            break
    int_ = reduce(lambda x, y: x * 10 + y, i11[0:i12:1])
    dec_ = 0.1 * reduce(lambda x, y: x * 0.1 + y, i11[-1:i12:-1])
    return int_ + dec_


DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': '.'}
print(str2float(s3))
print(float(s3))


# ##########filter过滤序列##########
# 去掉偶数
def is_odd(n):
    return n % 2 == 1


print("\n奇数为:")
print(list(filter(is_odd, [1, 2, 3, 4, 5, 6, 7])))
print(list(filter(lambda n: n % 2 == 1, [1, 2, 3, 4, 5, 6, 7])))


def not_empty(s):
    return s and s.strip()


print("\n对比filter和map:")
# filter只根据返回值确定是否保留
print(list(filter(not_empty, ['A   ', 'B ', None, ' C', '  '])))
# map会得到返回值
print(list(map(not_empty, ['A   ', 'B ', None, ' C', '  '])))

# 求素数
print("\n埃氏筛法求素数:")


# 生成从3开始的奇数
def odd_iter():
    n = 1
    while True:
        n += 2
        yield n
# [3, 5, 7, 9, 11, ...]


def not_divisible(n):
    return lambda x: x % n > 0


def primes():
    yield 2
    it = odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(not_divisible(n), it)


s4 = []
for n3 in primes():
    if n3 < 100:
        s4.append(n3)
    else:
        break
print(s4)


# 求回数
print("\n求回数:")


def odd_iter_n():
    n = 0
    yield n
    while True:
        n += 1
        yield n
# [0, 1, 2, 3, 4, ...]


def not_re():
    return lambda n: str(n) == str(n)[::-1]


def is_palindrome():
    it = odd_iter_n()
    while True:
        n = next(it)
        yield n
        it = filter(not_re(), it)


s4 = []
for n in is_palindrome():
    if n < 200:
        s4.append(n)
    else:
        break
print(s4)


# ##########排序##########
print("\n排序:")
s5 = [-21, 36, -12, 5, 9]
print(s5)
print(sorted(s5))
print(sorted(s5, key=lambda x: abs(x)))

s6 = ["amy", "Bob", "chilly", "Davy"]
print(sorted(s6))
print(sorted(s6, key=lambda x: str.lower(x)))
print(sorted(s6, key=lambda x: str.lower(x), reverse=True))


# ##########闭包##########
print("\n闭包")


def count():
    fs = []
    for i11 in range(1, 4):
        fs.append(lambda: i11*i11)
    return fs


f1, f2, f3 = count()
print(type(f1))
print(f1(), f2(), f3())


def count1():
    def f_(i11):
        return lambda: i11*i11
    fs = []
    for i12 in range(1, 4):
        fs.append(f_(i12))
    return fs


f1_, f2_, f3_ = count1()
print(type(f1_))
print(f1_(), f2_(), f3_())


# 递增整数
def add_c(k):
    def a_(i11):
        return lambda: i11
    fs = []
    for i12 in range(k):
        fs.append(a_(i12))
    return fs


f4_ = add_c(3)
for i in range(3):
    print(f4_[i]())


# 递增整数(答案)
def create_counter():
    x = 0

    def counter():
        nonlocal x
        x += 1
        return x
    return counter


counterA = create_counter()
print(counterA(), counterA(), counterA(), counterA(), counterA())


# ##########装饰器##########
print("\n装饰器decorator")


def log(func):
    def wrapper(*args_, **kw_):
        print('call %s()' % func.__name__)
        return func(*args_, **kw_)
    return wrapper


@log
def now():
    print("now")


now()
print(now.__name__)


# 传参日志
def log_(text):
    def decorator(func1):
        def wrapper1(*args1, **kw1):
            print('%s %s():' % (text, func1.__name__))
            return func1(*args1, **kw1)
        return wrapper1
    return decorator


@log_("execute")
def now1():
    print("now1")


print("\n传参decorator:")
now1()
print(now1.__name__)
now1 = log_("execute")(now1)
print(now1.__name__)

print("\n完整decorator")


def log2_(func):
    @functools.wraps(func)
    def wrapper(*args1, **kw1):
        print('call %s():' % func.__name__)
        return func(*args1, **kw1)
    return wrapper


@log2_
def now2():
    print("now2")


now2()
print(now2.__name__)


def log3_(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args1, **kw1):
            print('%s %s():' % (text, func.__name__))
            return func(*args1, **kw1)
        return wrapper
    return decorator


@log3_("execute")
def now3():
    print("now3")


now3()
print(now3.__name__)


# 打印函数执行时间
print("\n打印函数执行时间")


def timer(func):
    def wrapper(*args_):
        start = time.time()
        result = func(*args_)
        end = time.time()
        print("Total time:{:.3}s".format(end - start))
        return result
    return wrapper()


@timer
def time1():
    time.sleep(0.01)


# ##########偏函数##########
print("\n偏函数:")
print(int("12345"))
print(int("77", base=8))
print(int("eee", 16))
print(int("101011001100011", 2))

int2 = functools.partial(int, base=2)
print(int2("101011001100011"))

max10 = functools.partial(max, 10)
print(max10(5, 6, 7))


# ##########使用模块##########
print("\n使用模块")


def test():
    args2 = sys.argv
    if len(args2) == 1:
        print("Hello, %s!" % args2[0])
    else:
        print("Too many arguments!")


if __name__ == "__main__":
    test()

# ##########命名##########
print('''
(1)、以单下划线开头，表示这是一个保护成员，只有类对象和子类对象自己能访问到这些变量。
以单下划线开头的变量和函数被默认当作是内部函数，使用from module improt *时不会被获取，但是使用import module可以获取

(2)、以单下划线结尾仅仅是为了区别该名称与关键词

(3)、双下划线开头，表示为私有成员，只允许类本身访问，子类也不行。在文本上被替换为_class__method

(4)、双下划线开头，双下划线结尾。一种约定，Python内部的名字，用来区别其他用户自定义的命名,以防冲突。
是一些 Python 的“魔术”对象，表示这是一个特殊成员，例如：定义类的时候，若是添加__init__方法，
那么在创建类的实例的时候，实例会自动调用这个方法，一般用来对实例的属性进行初使化，
Python不建议将自己命名的方法写为这种形式。
''')


# ##########面向对象编程##########
class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print("%s: %s" % (self.name, self.score))


AAA = Student("name AAA", 23)
BBB = Student("name BBB", 24)
AAA.print_score()
BBB.print_score()
AAA.a = 'a'
print(AAA.a)


# 访问限制
class Student2(object):

    def __init__(self, name, score):
        self.__name = name
        self.__score = score

    def print_score(self):
        print("%s: %s" % (self.__name, self.__score))


CCC = Student2("name CCC", 25)
CCC.print_score()


# ##########继承和多态##########
print("\n继承和多态：")


class Animal(object):
    def run(self):
        print("Animal is running...")


class Dog(Animal):
    pass


dog = Dog()
dog.run()


# 子类run覆盖父类run
class Cat(Animal):
    def run(self):
        print("Cat is running...")


cat = Cat()
cat.run()

print(isinstance(cat, Animal))
print(isinstance(cat, Dog))
print(isinstance(cat, Cat))


def run_twice(animal):
    animal.run()
    animal.run()


run_twice(Animal())
run_twice(Cat())

# ##########获取对象信息##########
# type
print("\n获取对象信息：")
print(type(abs))
print(type(123))
print(type('a') == str)


def fn():
    pass


# isinstance
print(isinstance(fn, types.FunctionType))
print(isinstance(abs, types.BuiltinMethodType))
print(isinstance(lambda x: x, types.LambdaType))
print(isinstance((x for x in range(10)), types.GeneratorType))
print(isinstance(dog, Dog))
print(isinstance(dog, Animal))
print(isinstance(dog, Cat))
print(isinstance([1, 2, 3], (list, tuple)))
print(isinstance((1, 2, 3), (list, tuple)))

# dir:获取所有属性和方法
print(dir("abc"))
print(dir(123))
print(len("aaa"))
print("aaa".__len__())
# hasattr:判断是否存在属性或方法
print(hasattr(Animal, 'run'))
# setattr:设置属性
print(hasattr(Dog, 'weight'))
setattr(Dog, 'weight', 20)
print(hasattr(Dog, 'weight'))
# getattr:获取属性
print(getattr(Dog, 'weight'))
# 获取不到返回默认值
print(getattr(Dog, 'no', 404))
run_cat = getattr(cat, 'run')
run_cat()


# 给实例绑定方法
def fly(self, high):
    self.high = high


bird = Animal()
bird.fly = types.MethodType(fly, bird)
bird.fly(20)
print(bird.high)


# 给class绑定方法
def jump(self, h):
    self.h = h


Animal.jump = jump
pig = Animal()
pig.jump(10)
print(pig.h)


# __slots__限制实例属性
class Person(object):
    __slots__ = ("name", "age")


p = Person()
p.name = "Ppp"
p.age = 24
# p.gender = "Male" AttributeError: 'Person' object has no attribute 'gender'
print(p.name, p.age)


class PPerson(Person):
    pass


# __slots__对继承子类不起作用，除非子类也定义__slots__
pp = PPerson()
PPerson.gender = "Male"
print(pp.gender)


# @property将方法变为属性调用
class PPPerson(object):
    def __init__(self):
        self._score = None
        self._birth = None

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError("score mast be an integer!")
        if value < 0 or value > 100:
            raise ValueError("score mast between 0 ~ 100!")
        self._score = value

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return time.localtime().tm_year - self._birth


print("\n@property将方法变为属性调用")
s = PPPerson()
# s.score = 999 报错
s.score = 100
s.birth = 1998
# s.age = 29 报错
print(s.score, s.birth, s.age)


# ##########多重继承##########
class Animals(object):
    pass


class Mammal(Animals):
    pass


class Birds(Animals):
    pass


class Runnable(object):
    @staticmethod
    def run():
        print("Running...")


class Flyable(object):
    @staticmethod
    def fly():
        print("Flying...")


class Human(Mammal, Runnable):
    pass


class Monkey(Mammal, Runnable):
    pass


class Parrot(Birds, Flyable):
    pass


class Crow(Birds, Flyable):
    pass


cc = Crow()
cc.fly()


# ##########定制类##########
print("\n定制类:")


class NotDing(object):
    def __init__(self, name):
        self.name = name


n = NotDing('n')
print(n)


class Ding(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Ding object (name=%s)' % self.name
    __repr__ = __str__


d = Ding('d')
print(d)


class FibIter(object):
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 200:
            raise StopIteration()
        return self.a


for n in FibIter():
    print(n)


class FibList(object):
    def __getitem__(self, item):
        if isinstance(item, int):
            a, b = 1, 1
            for x in range(item):
                a, b = b, a + b
            return a
        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            if start is None:
                start = 0
            a, b = 1, 1
            li = []
            for x in range(stop):
                if x >= start:
                    li.append(a)
                a, b = b, a + b
            return li


f = FibList()
print(f[0:10])
print(f[50])


# 不存在的属性
class BuCZ(object):
    def __init__(self):
        self.a = "存在"

    def __getattr__(self, attr):
        if attr != "a":
            return "don't have '%s' property" % attr

    def __call__(self):
        print("测试%s" % self.a)


ppp = BuCZ()
print(ppp.a)
print(ppp.b)
ppp()

# 用callable判断是否可调用
print(callable(Student))
print(callable(FibIter))
print(callable(FibList))
print(callable(BuCZ))
print(callable("str"))

# ##########枚举类##########
print("\n枚举类:")
E = Enum("Week", ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
p = E.Mon
print(p)
for name, member in E.__members__.items():
    print(name, "->", member, ',', member.value)


# 自定义枚举类
@unique  # 保证没有重复值
class Weekday(Enum):
    Mon = 0
    Tue = 1
    Wed = 2
    Thu = 3
    Fri = 4
    Sat = 5
    Sun = 6


day1 = Weekday.Mon
print("day1:\t\t\t\t", day1)
print("Weekday.Mon:\t\t", Weekday.Mon)
print("Weekday.Mon.value:\t", Weekday.Mon.value)
print("Weekday(1):\t\t\t", Weekday(0))
for name, member in Weekday.__members__.items():
    print(name, "->", member, ',', member.value)


# ##########元类##########
def fn(_, name="world"):
    print("Hello %s!" % name)


Hello = type("Hello", (object,), dict(hello=fn))
h = Hello()
h.hello()


# ##########错误处理##########
# print("\n错误处理:")
# try:
#     print('try...')
#     r = 10 / 0
#     print('result:', r)
# except ZeroDivisionError as e:
#     print('except:', e)
#     logging.exception(e)  # 记录并继续执行
# finally:
#     print('finally...')
# print('END')

# ##########文件读写##########
print("\n文件读写:")
# 全部读取
with open('test.txt', 'r', encoding="utf-8") as f:
    print(f.read())

# 按行读取
with open('test.txt', 'r', encoding="utf-8") as f:
    for line in f.readlines():
        print(line.strip())

# 重写文件
with open('test.txt', 'w', encoding="utf-8") as f:
    f.write("Hello World!-1\n")

# 追加写文件
with open('test.txt', 'a', encoding="utf-8") as f:
    f.write("Hello World!-2")

# StringIO和BytesIO
f = StringIO()
f.write("StringIO")
print(f.getvalue())

f = StringIO("StringIO!\nStringIO?\nStringIO!")
while True:
    s = f.readline()
    if s == '':
        break
    print(s.strip())

f = BytesIO()
f.write("中文".encode("utf-8"))
print(f.getvalue())
f = BytesIO(b"\xe4\xb8\xad\xe6\x96\x87")
print(f.getvalue())

# ##########操作文件和目录##########
print("\n操作文件和目录:")
print(os.name)
print(os.environ)
print(os.environ.get("PATH"))

# ##########序列化##########
di = dict(name="Bob", age=20, score=90)
f = open("dump.txt", "wb")
pickle.dump(di, f)
f.close()
f = open("dump.txt", "rb")
do = pickle.load(f)
f.close()
print(do)

# json
di = dict(name="Bob", age=20, score=90)
print(json.dumps(di))
json_str = "{\"name\": \"Bob\", \"age\": 20, \"score\": 90}"
print(json.loads(json_str))

# ##########线程和进程##########
print("\n线程和进程:")
print("Process (%s) start..." % os.getpid())

# ##########正则表达式##########
print("\n正则表达式:")
print("\"\\d\"匹配数字:", re.match(r"00\d", "007"))
print("\"\\d\"匹配字母:", re.match(r"00\d", "00w"))
print("\"\\w\"匹配字母:", re.match(r"00\w", "00waa"))
print("\"\\d\\w\"组合:", re.match(r"\w\d\w", "B2B"))
print("\".\"匹配任意字符:", re.match(r"...", "试一试abc"))
print("\"\\s\"匹配空格:", re.match(r"\s", "    "))
print("\"*\"表示任意个字符:", re.match(r"\d*", "123456789"))
print("\"*\"表示任意个字符:", re.match(r"\d*", ""))
print("\"+\"表示至少一个字符:", re.match(r"\d+", "123456789"))
print("\"+\"表示至少一个字符:", re.match(r"\d+", ""))
print("\"?\"表示0~1个字符:", re.match(r"\d?", "123"))
print("\"?\"表示0~1个字符:", re.match(r"\d?", "123"))
print("\"{n}\"表示n个字符:", re.match(r"\d{3}", "12345"))
print("\"{n,m}\"表示n~m个字符:", re.match(r"\d{3,4}", "12345"))
print("\"A|B\"表示匹配A或B:", re.match("(p|P)ython", "python"))
print("匹配电话号码:", re.match(r"\d{3}-\d{8}", "010-12345678"))
print("\"[]\"表示范围:", re.match(r"[0-9a-zA-z_]*", "_1a2B3c4D"))
print("\"[]\"表示范围:", re.match(r"[0-9a-zA-z_]+", "_1a2B3c4D"))
print("python合法变量:", re.match(r"[a-zA-z_][0-9a-zA-z_]{0,19}", "_str"))
print("python合法变量:", re.match(r"[a-zA-z_][0-9a-zA-z_]{0,19}", "1aa"))
print("\"^\"表示开头:", re.match(r"^_[0-9a-zA-z_]*", "_1aa3s"))
print("\"$\"表示结尾:", re.match(r"([0-9a-zA-z_]*_$)", "1a4j_"))
# 分割字符串
print(list("a  b   c ".strip(' ')))
print(re.split(r'\s+', "a,b, c    d"))
print(re.split(r'[\s,]+', "a,b, c    d"))
print(re.split(r'[\s,;]+', "a,b, ;;c ;   d"))
# 分组
Tel_number = re.match(r"^(\d{3})-(\d{8})", "010-12345678")
print(Tel_number.group(0))
print(Tel_number.group(1))
print(Tel_number.group(2))
# 预编译
re_tele = re.compile(r"^(\d{3})-(\d{8})")
print(re_tele.match("010-12345678").groups())
