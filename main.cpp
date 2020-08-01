#include <iostream>
#include <vector>
#include "make_dataset.h"
#include "mlp.h"

int main(void)
{
	vector<vector<double>> input_gate = make_dataset::input_set(2);
	vector<double> temp_output = make_dataset::output_set(input_gate, 2);
	vector<vector<double>> output_gate;
	for (int i = 0; i < input_gate.size(); i++)
	{
		output_gate.push_back(vector<double>(1, temp_output[i]));
	}
	vector<vector<double>> input_donut = {
	{0.,0.},
	 {0.,1.},
	 {1.,0.},
	 {1.,1.},
	 {0.5,1.},
	 {1.,0.5},
	 {0.,0.5},
	 {0.5,0.},
	 {0.5,0.5} };
	vector<vector<double>> output_donut =
	{
		 {0}, {0},{0},{0}, {0},{0},{0},{0},{1}
	};
	vector<vector<double>> input_donut_2 = {
		{0.,0.},	
		{0.5,0.5},
		{0.,1.},
		{1.,0.},
		{1.,1.},
		{0.5,0.5},
		{0.5,1.},
		{1.,0.5},
		{0.,0.5},
		{0.5,0.},
		{0.5,0.5}
	};
	vector<vector<double>> output_donut_2 =
	{
		 {0},{1}, {0},{0}, {0},{1}, {0},{0},{0},{0},{1}
	};
	mlp a(2, 1, 1,2 ,0.5, 0.00005, input_donut, output_donut);//inputdim,outputdim,layernum,nodenum,learning_rate,tolerance,input,output
	a.learning();
	while (1);
}
