function result = map(list1,list2,n)
	[B1,I1] = sort(list1,'descend');
	[B2,I2] = sort(list2,'descend');
	result = 0;
	tmp = 0.0;
	for i=1:n
		tmp = 0.0;
		for j=1:i
			if I1(j) == I2(i)
				tmp = tmp + 1.0;
			end
		end
		result = result + tmp / i;
	end
	result = result/n;
end