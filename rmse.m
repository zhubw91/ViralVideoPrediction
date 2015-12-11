function result = rmse(list1,list2)
	result = sqrt(mean((list1 - list2).*(list1 - list2)));
end