#pragma once
#include <vector>
#include <ctime>
#include <fstream>
#include <string>
#include "matrix.h"
using namespace std;
/**
	multi layer perceptron�� ���� �н��ϱ� ���� ��� ������ �Լ��� ���Ե� class�Դϴ�.
*/
class mlp
{
private:
	int input_dim;//input�� ����
	int output_dim;//ouput�� ����
	int layer_num;//hidden layer�� ��
	int node_num;//layer�� node�� ��
	double learning_rate;
	double tolerance;
	vector<vector<vector<double>>> old_weight;//back propagation ���� weight
	vector<vector<vector<double>>> new_weight;//back propagation ���� weight
	vector<vector<double>> node;//�� layer�� activate_function(net)�� ����.
	vector<vector<double>> delta;//delta���� ����.
	vector<vector<double>> input;//input data ����
	vector<vector<double>> output;//output data����
	vector<vector<double>> bias;//bias�� ����
	vector<vector<double>> bias_weight;//bias weight�� ����.
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
	*@details: ������ open���ֱ� ���� �޼ҵ�
	*@param args:
	*@return
	*/
	void init_file();
	/**
	*@method name: save_*
	*@details: back progation ���� �� ,�ٲٱ� ���� �ٲ� ���� hidden layer output, weight, bias weight���� ���Ͽ� �����ϴ� �Լ�.
	*@param args: �����ϴ� ������ epoch, � input���� ���� �н��ϴ��� �˷��ִ� error_num
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
	*@details:	forward propagation�� output�� ��ȯ�ϴ� �Լ�.
	*@param args: forward propagation�� output, � input���� ���� �н��ߴ��� �˷��ִ� data_num
	*@return void
	*/
	double error_cal(vector<double> cal_output, int data_num);
	/**
	*@method name: activate_function
	*@details:	sigmoid�Լ�
	*@param args: net ����� ��
	*@return void
	*/
	double activate_function(double net);
	/**
	*@method name: activate_function_drv
	*@details:	activate function �̺��� ��
	*@param args: net ����� ��
	*@return void
	*/
	double activate_function_drv(double net);
	/**
	*@method name: forward propagation
	*@details: ������ �ִ� data�� ���� forward propagation�� �ǽ���.
	*@param args: � input���� ���� �н��ϴ��� �˷��ִ� data_num
	*@return void
	*/
	void forward_propagation(int data_num);
	void forward_propagation_each_node(int data_num, vector<vector<double>> &temp_node);
	/**
	*@method name: set_back_delta
	*@details: ������ �ִ� data�� ���� foward propagation�� �� ��, ���� back propagation�� �ʿ��ϴٸ� delta�� �����.
	*@param args: � input���� ���� �н��ϴ��� �˷��ִ� data_num
	*@return void
	*/
	void set_back_delta(int data_num);
	/**
	*@method name: back_propagation
	*@details: ������ �ִ� data�� ���� back_propagation�� ��.
	*@param args:
	*@return void
	*/
	void back_propagation();
	void for_next();

public:
	mlp(int input_dim, int output_dim, int layer_num, int node_num, double learning_rate, double tolerance, vector<vector<double>> input, vector<vector<double>> output);
	void learning();
};

