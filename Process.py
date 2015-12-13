import math
f = open("realdata_all.csv")

# def discrete(x):
#     k = 10
#     if x >= 10:
#         return 10
#     elif x <= 0:
#         return 0
#     else:
#         delta = math.log(10) / k
#         return int(round(math.log(x+1)/delta))

def discrete(x):
    k = 10
    if x >= 100:
        return 9
    elif x <= 0:
        return 0
    else:
        delta = 100 / k
        return int(round(x / delta))

result = []
output = open('traindata_old_scale.csv','w')
for line in f:
    s = line.split(',')
    s = [int(x) for x in s]
    # if s[0] < 10000:
    #     continue
    tmp = []
    flag = 0
    for index,x in enumerate(s):
        if index > 0:
            if s[index-1] == 0:
                tmp.append(discrete(s[index]))
            else:
                if ( (s[index]+0.0) / s[index-1] ) < 1:
                    flag = 1
                    break
                tmp.append( discrete((((s[index]+0.0)/s[index-1]) - 1)*100) )
    if flag == 1:
        continue
    tmp = [str(x) for x in tmp]
    output.write(','.join(tmp))
    output.write('\n')





