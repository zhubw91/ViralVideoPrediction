function avg_map = hmm(x)
load traindata_old_all.csv;
load realdata_all.csv;
traindata_old = traindata_old_all;
realdata_origin = realdata_all;

train_size = floor(size(traindata_old,1)/10 * 9);
test_size = size(traindata_old,1) - train_size - 1;
date_size = 16;
date_train_size = date_size-1;
data_origin = traindata_old + 1;
state_num = x;
ob_state_num = 11;

k = 5;
map_result = zeros(k,4);
% Cross Validation
for t=1:k
    q = randperm(size(data_origin,1));
    data = data_origin(q,:);
    realdata = realdata_origin(q,:);
    trainset = data(1:train_size,1:date_train_size);
    testset = data(train_size+1:train_size+test_size,:);

    trans = ones(state_num,state_num);
    for i=1:state_num
        for j=1:state_num
            if j < i
                trans(i,j) = 1.5-0.1*(i-j);
            elseif j > i
                trans(i,j) = 1.5-0.1*(j-i);
            end 
        end
    end 
    trans = trans./(sum(trans,2)*ones(1,state_num));
    emis = ones(state_num,ob_state_num)./ob_state_num;
    trans
    [estTR,estE] = hmmtrain(trainset,trans,emis);

    cnt = 0;
    correct = 0;
    test_real = realdata(train_size+1:train_size+test_size,:);
    result = zeros(1,test_size);
    for i = 1:test_size
        pstate = hmmviterbi(testset(i,1:date_train_size),estTR,estE);
        current_state = pstate(date_train_size);
        [pre_max,pre_state] = max(estTR(current_state,:));
        [pre_max,index] = max(estE(pre_state,:));
        result(i) = index;
        if(index == testset(i,date_size))
            correct = correct + 1;
        end
        cnt = cnt + 1;   
    end

    standard = ones(1,test_size);
    pre = ones(1,test_size);
    for i = 1:test_size
        % standard(i) = test_real(i,date_size+1);
        standard(i) = test_real(i,date_size+1) - test_real(i,date_size);
        % pre(i) = 0;
        % pre(i) = test_real(i,date_size-1) - test_real(i,date_size-2);
        pre(i) = floor(test_real(i,date_size) * (1+(exp(result(i)*(log(100)-1) / 10) /100))) - test_real(i,date_size);
        % pre(i) = floor(test_real(i,date_size) * (1+(exp(result(i)*log(20) / 10) /100)));
    end
    standard
    pre
    map_result(t,1) = map(standard, pre, 1);
    map_result(t,2) = map(standard, pre, 5);
    map_result(t,3) = map(standard, pre, 10);
    map_result(t,4) = map(standard, pre, 20);

end

avg_map = mean(map_result,1);
estTR
estE

