#pragma once
#include <vector>
#include <cassert>
using namespace std;
class make_dataset
{
public:
	static vector<vector<double>> input_set(int dim); //dim의 값을 가지고 input값을 만들어 배열에 담아 반환.ex) dim = 2 -> 11,10,01,00이 담긴 2차원 배열 반환
	static vector<double> output_set(vector<vector<double>> input, int type);//input값과 연산의 종류를 가지고 알맞은 output을 계산.
																       //type1->and, type=2->or type=3->xor
																	   //ex)위 주석과 같은 input배열이 주어지고 type1이 1이라면, 1,0,0,0이 담긴 1차원 배열 반환
};

