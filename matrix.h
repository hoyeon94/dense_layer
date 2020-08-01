#pragma once
#include <vector>
#include <iostream>
//matrix 연산과 console 출력을 위한 class
using namespace std;
class matrix
{
public:
	static vector<double> scalar(vector<double> a, double b);
	static double dot(vector<double> a, vector<double> b);
	static vector<double> add(vector<double> a, vector<double> b);
	static vector<double> mul(vector<double> a, vector<double> b);
	static double self_add(vector<double> a);
	static void print_one_dim_matrix(vector<double> mat);
	static void print_three_dim_matrix(vector<vector<vector<double>>> mat);
	static void print_two_dim_matrix(vector<vector<double>> mat);
};

