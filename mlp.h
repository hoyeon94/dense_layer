#pragma once
#include <vector>
#include <ctime>
#include <fstream>
#include <string>
#include "matrix.h"
using namespace std;
/**
	multi layer perceptron을 통해 학습하기 위한 모든 변수와 함수가 포함된 class입니다.
*/
class mlp
{
private:
	int input_dim;//input의 개수
	int output_dim;//ouput의 개수
	int layer_num;//hidden layer의 수
	int node_num;//layer당 node의 수
	double learning_rate;
	double tolerance;
	vector<vector<vector<double>>> old_weight;//back propagation 전의 weight
	vector<vector<vector<double>>> new_weight;//back propagation 후의 weight
	vector<vector<double>> node;//각 layer의 activate_function(net)을 저장.
	vector<vector<double>> delta;//delta값을 저장.
	vector<vector<double>> input;//input data 저장
	vector<vector<double>> output;//output data저장
	vector<vector<double>> bias;//bias값 저장
	vector<vector<double>> bias_weight;//bias weight값 저장.
	ofstream *node_file;
	ofstream *new_weight_file;
	ofstream *old_weight_file;
	ofstream *error_file;
	ofstream *old_node_file;
	ofstream *old_bias_file;
	ofstream *new_bias_file;
	void init_bias();
	void random_init_weight();
	void set_bias(vector<vector<double>> bias);
	/**
	*@method name:init_file
	*@details: 파일을 open해주기 위한 메소드
	*@param args:
	*@return
	*/
	void init_file();
	/**
	*@method name: save_*
	*@details: back progation 전과 후 ,바꾸기 전과 바꾼 후의 hidden layer output, weight, bias weight등을 파일에 저장하는 함수.
	*@param args: 저장하는 시점의 epoch, 어떤 input값에 대해 학습하는지 알려주는 error_num
	*@return void
	*/
	void save_new_bias(int epoch, int error_num);
	void save_old_bias(int epoch, int error_num);
	void save_new_weight(int epoch, int error_num);
	void save_old_weight(int epoch, int error_num);
	void save_error_file(int epoch, int error_cnt);
	void save_node_file(int epoch, int errored_input);
	void save_old_node_file(int epoch, int errored_input);
	/**
	*@method name: error_cal
	*@details:	forward propagation의 output을 반환하는 함수.
	*@param args: forward propagation의 output, 어떤 input값에 대해 학습했는지 알려주는 data_num
	*@return void
	*/
	double error_cal(vector<double> cal_output, int data_num);
	/**
	*@method name: activate_function
	*@details:	sigmoid함수
	*@param args: net 계산한 값
	*@return void
	*/
	double activate_function(double net);
	/**
	*@method name: activate_function_drv
	*@details:	activate function 미분한 값
	*@param args: net 계산한 값
	*@return void
	*/
	double activate_function_drv(double net);
	/**
	*@method name: forward propagation
	*@details: 가지고 있는 data에 대해 forward propagation을 실시함.
	*@param args: 어떤 input값에 대해 학습하는지 알려주는 data_num
	*@return void
	*/
	void forward_propagation(int data_num);
	void forward_propagation_each_node(int data_num, vector<vector<double>> &temp_node);
	/**
	*@method name: set_back_delta
	*@details: 가지고 있는 data에 대해 foward propagation을 한 뒤, 만약 back propagation이 필요하다면 delta를 계산함.
	*@param args: 어떤 input값에 대해 학습하는지 알려주는 data_num
	*@return void
	*/
	void set_back_delta(int data_num);
	/**
	*@method name: back_propagation
	*@details: 가지고 있는 data에 대해 back_propagation을 함.
	*@param args:
	*@return void
	*/
	void back_propagation();
	void for_next();

public:
	mlp(int input_dim, int output_dim, int layer_num, int node_num, double learning_rate, double tolerance, vector<vector<double>> input, vector<vector<double>> output);
	void learning();
};

