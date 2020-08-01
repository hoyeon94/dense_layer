#include "make_dataset.h"
vector<vector<double>> make_dataset::input_set(int dim)
{
	vector<vector<double>> ret;
	int data_count = pow(2, dim);
	for (int data_num = 0; data_num < data_count; data_num++)
	{
		ret.push_back(vector<double>(0, 0));
		unsigned int number = data_num;
		for (int i = 0; i < dim; i++)
		{
			ret[data_num].push_back(number % 2);
			number >>= 1;
		}
	}
	return ret;
}
vector<double> make_dataset::output_set(vector<vector<double>> input, int type)
{
	vector<double> ret;
	switch (type)
	{
	case 1:
		for (vector<double> each_input : input)
		{
			bool sum = each_input[0];
			for (int i = 1; i < each_input.size(); i++)
			{
				sum = sum & (int)each_input[i];
			}
			ret.push_back(sum);
		}
		break;
	case 2:
		for (vector<double> each_input : input)
		{
			bool sum = each_input[0];
			for (int i = 1; i < each_input.size(); i++)
			{
				sum = sum | (int)each_input[i];
			}
			ret.push_back(sum);
		}
		break;
	case 3:
		for (vector<double> each_input : input)
		{
			bool sum = each_input[0];
			for (int i = 1; i < each_input.size(); i++)
			{
				sum = sum ^ (int)each_input[i];
			}
			ret.push_back(sum);
		}
		break;
	default:
		assert(0 && "incorrect type"); //만약 1,2,3 이외의 값이 입력되었다면 프로그램 종료.
	}
	return ret;
}