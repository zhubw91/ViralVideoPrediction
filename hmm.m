load traindata2.csv;
level_num = 9;
max_cnt = max(max(traindata2));
min_cnt = min(min(traindata2));
delta = (max_cnt - min_cnt)/level_num;
data = fix(traindata2./delta) + 1;
trainset = data(1:400,:);
testset = data(401:472,:);
trans = [0.5,0.5;
      0.5,0.5];
emis = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

[estTR,estE] = hmmtrain(trainset,trans,emis);
cnt = 0;
correct = 0;
for i = 1:72
    pstate = hmmdecode(testset(i,1:4),estTR,estE);
    if(isnan(pstate(1,4)))
        continue;
    end
    [m,index] = max(estTR(round(pstate(1,4)),:));
    [m,index] = max(estE(index,:));
    if(index == testset(i,5))
        correct = correct + 1;
    end
    cnt = cnt + 1;   
end

display correct,cnt




