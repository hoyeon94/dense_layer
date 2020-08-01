#pragma once
#include <vector>
#include <cassert>
using namespace std;
class make_dataset
{
public:
	static vector<vector<double>> input_set(int dim); //dim�� ���� ������ input���� ����� �迭�� ��� ��ȯ.ex) dim = 2 -> 11,10,01,00�� ��� 2���� �迭 ��ȯ
	static vector<double> output_set(vector<vector<double>> input, int type);//input���� ������ ������ ������ �˸��� output�� ���.
																       //type1->and, type=2->or type=3->xor
																	   //ex)�� �ּ��� ���� input�迭�� �־����� type1�� 1�̶��, 1,0,0,0�� ��� 1���� �迭 ��ȯ
};

