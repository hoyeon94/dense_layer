#include "matrix.h"
vector<double> matrix::scalar(vector<double> a, double b)
{
	vector<double> ret;
	for (int i = 0; i < a.size(); i++)
	{
		ret.push_back(a[i] * b);
	}
	return ret;
}
double matrix::dot(vector<double> a, vector<double> b)
{
	double ret = 0;
	for (int i = 0; i < a.size(); i++)
	{
		ret += a[i] * b[i];
	}
	return ret;
}
vector<double> matrix::mul(vector<double> a, vector<double> b)
{
	vector<double> ret;
	for (int i = 0; i < a.size(); i++)
	{
		ret.push_back(a[i]* b[i]);
	}
	return ret;
}
vector<double> matrix::add(vector<double> a, vector<double> b)
{
	vector<double> ret;
	for (int i = 0; i < a.size(); i++)
	{
		ret.push_back(a[i] + b[i]);
	}
	return ret;
}
double matrix::self_add(vector<double> a)
{
	double ret = 0;
	for (int i = 0; i < a.size(); i++)
	{
		ret += a[i];
	}
	return ret;
}

void matrix::print_one_dim_matrix(vector<double> mat)
{
	for (int i = 0; i < mat.size(); i++)
	{
		std::cout << mat[i] << " ";
	}
	std::cout << std::endl;
}
void matrix::print_three_dim_matrix(vector<vector<vector<double>>> mat)
{
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[i].size(); j++)
		{
			for (int z = 0; z < mat[i][j].size(); z++)
			{
				std::cout << mat[i][j][z] << " ";
			}
			std::cout << "  ><   ";
		}
		std::cout << std::endl;
	}
}
void matrix::print_two_dim_matrix(vector<vector<double>> mat)
{
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[i].size(); j++)
		{
			std::cout << mat[i][j] << " ";
		}
		std::cout << std::endl;
	}
}