
f = open("../../source.text","r") 
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
line = lines[0]
print(line)
print(len(line))