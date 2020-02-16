## 1. Introduction to Computer Components ##

print("Hello World!")

## 2. Data Storage, Memory, and RAM ##

my_int = 90
int_addr = id(my_int)
my_str = "Hello"
str_addr = id(my_str)

## 4. Understanding How Python Stores Data ##

import sys

my_int = 200
size_of_my_int = sys.getsizeof(my_int)

int1 = 10
int2 = 100000
str1 = "Hello"
str2 = "Hi"

int_diff = sys.getsizeof(int1) - sys.getsizeof(int2)
str_diff = sys.getsizeof(str1) - sys.getsizeof(str2)

print(int_diff)
print(str_diff)

## 6. Understanding Disk Storage ##

import time
import csv

before_file_read = time.clock()
f = open("list.csv", "r")
list_from_file = list(csv.reader(f))
after_file_read = time.clock()
before_string_parse = time.clock()
list_from_RAM = "1,2,3,4,5,6,7,8,9,10".split(",")
after_string_parse = time.clock()

file_time = after_file_read - before_file_read
RAM_time = after_string_parse - before_string_parse

## 8. An Overview of Binary ##

num1 = 0*(2**3) + 1*(2**2) + 1*(2**1) + 0*(2**0)
num2 = 1*(2**3) + 0*(2**2) + 0*(2**1) + 1*(2**0)
num3 = 1*(2**5) + 0*(2**4) + 0*(2**3) + 1*(2**2) + 0*(2**1) + 0*(2**0)

## 9. Computation and Control Flow ##

a = 5
b = 10
print("On line 3")
if a == 5:
    print("On line 5")
else:
    print("On line 7")
if b < a:
    print("On line 9")
elif b == a:
    print("On line 11")
else:
    for i in range(3):
        print("On line 14")

printed_lines = [3, 5, 14, 14, 14]

## 10. Functions in Memory ##

def my_func():
    print("On line 2")
a = 5
b = 10
print("On line 5")
my_func()
print("On line 7")
my_func()

printed_lines = [5, 2, 7, 2]