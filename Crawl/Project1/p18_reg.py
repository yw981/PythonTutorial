import re


# 正则表达式以 r 引导
reg=r"x[^ab0-9]y"
# "^"出现在[]的第一个字符位置，就代表取反，例如[^ab0-9]表示不是 a、 b，也不是0-9 的数字。
m=re.search(reg,"xayx2yxcy")
print(m)

s="I am testing search function"
reg=r"[A-Za-z]+\b"
m=re.search(reg,s)
while m!=None:
    start=m.start()
    end=m.end()
    print(s[start:end])
    s=s[end:]
    m=re.search(reg,s)