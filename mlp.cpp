#include "mlp.h"
mlp::mlp(int input_dim, int output_dim, int layer_num, int node_num, double learning_rate, double tolerance, vector<vector<double>> input, vector<vector<double>> output)
//input�� ����, output�� ����, hiddenlayer�� ����, hidden layer �ϳ��� ����ǰ���, learning rate, tolerance, input,ouputdata�� ���ʴ�� �޾Ƽ� Ŭ������ �ʱ�ȭ�Ѵ�.
{
	srand((unsigned int)time(NULL));
	this->input_dim = input_dim;
	this->output_dim = output_dim;
	this->layer_num = layer_num;
	this->node_num = node_num;
	this->input = input;
	this->output = output;
	this->learning_rate = learning_rate;
	this->tolerance = tolerance;
	old_weight = vector<vector<vector<double>>>(layer_num + 1, vector<vector<double>>(node_num, vector<double>(node_num, 0)));
	node = vector<vector<double>>(layer_num + 2, vector<double>(node_num, 0));
	delta = vector<vector<double>>(layer_num + 1, vector<double>(node_num, 0));
	random_init_weight();
	new_weight = old_weight;
	init_bias();
	init_file();
}
void mlp::init_file()
{
	string now_time = to_string(time(NULL));
	string target_old_weight = "D:\\old_weight";
	target_old_weight.append(now_time);
	target_old_weight.append(".txt");
	string target_new_weight = "D:\\new_weight";
	target_new_weight.append(now_time);
	target_new_weight.append(".txt");
	string target_error = "D:\\error";
	target_error.append(now_time);
	target_error.append(".txt");
	string target_node = "D:\\node";
	target_node.append(now_time);
	target_node.append(".txt");
	string target_old_node = "D:\\old_node";
	target_old_node.append(now_time);
	target_old_node.append(".txt");
	string target_old_bias = "D:\\old_bias";
	target_old_bias.append(now_time);
	target_old_bias.append(".txt");
	string target_new_bias = "D:\\new_bias";
	target_new_bias.append(now_time);
	target_new_bias.append(".txt");
	new_weight_file = new ofstream(target_new_weight);
	old_weight_file = new ofstream(target_old_weight);
	node_file = new ofstream(target_node);
	old_node_file = new ofstream(target_old_node);
	error_file = new ofstream(target_error);
	old_bias_file = new ofstream(target_old_bias);
	new_bias_file = new ofstream(target_new_bias);
}
void mlp::save_new_bias(int epoch, int error_num)
{
	*new_bias_file << epoch << std::endl;
	*new_bias_file << input[error_num][0] << " " << input[error_num][1] << std::endl;
	*new_bias_file << bias[0][0] * bias_weight[0][0] << " " << bias[0][1] * bias_weight[0][1] << std::endl;
	*new_bias_file << bias[1][0] * bias_weight[1][0] << std::endl;
}
void mlp::save_old_bias(int epoch, int error_num)
{
	*old_bias_file << epoch << std::endl;
	*old_bias_file << input[error_num][0] << " " << input[error_num][1] << std::endl;
	*old_bias_file << bias[0][0] * bias_weight[0][0] << " " << bias[0][1] * bias_weight[0][1] << std::endl;
	*old_bias_file << bias[1][0] * bias_weight[1][0] << std::endl;
}
void mlp::save_new_weight(int epoch, int error_num)
{
	*new_weight_file << epoch << std::endl;
	*new_weight_file << input[error_num][0] << " " << input[error_num][1] << std::endl;
	vector<vector<vector<double>>> mat = new_weight;
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[i].size(); j++)
		{
			for (int z = 0; z < mat[i][j].size(); z++)
			{
				*new_weight_file << mat[i][j][z] << " ";
			}
			*new_weight_file << " ";
		}
		*new_weight_file << std::endl;
	}
}
void mlp::save_old_weight(int epoch, int error_num)
{
	*old_weight_file << epoch << std::endl;
	*old_weight_file << input[error_num][0] << " " << input[error_num][1] << std::endl;
	vector<vector<vector<double>>> mat = old_weight;
	for (int i = 0; i < mat.size(); i++)
	{
		for (int j = 0; j < mat[i].size(); j++)
		{
			for (int z = 0; z < mat[i][j].size(); z++)
			{
				*old_weight_file << mat[i][j][z] << " ";
			}
			*old_weight_file << " ";
		}
		*old_weight_file << std::endl;
	}
}
void mlp::save_error_file(int epoch, int error_cnt)
{
	*error_file << epoch << std::endl;
	*error_file << error_cnt << std::endl;
}
void mlp::save_node_file(int epoch, int errored_input)
{
	vector<vector<double>> temp_node = this->node;
	*node_file << epoch << std::endl;
	*node_file << input[errored_input][0] << " " << input[errored_input][1] << std::endl;
	for (int i = 0; i < input.size(); i++)
	{
		forward_propagation_each_node(i, temp_node);
		*node_file << temp_node[1][0] << " " << temp_node[1][1] << " " << std::endl;
	}
}
void mlp::save_old_node_file(int epoch, int errored_input)
{
	vector<vector<double>> temp_node = this->node;
	*old_node_file << epoch << std::endl;
	*old_node_file << input[errored_input][0] << " " << input[errored_input][1] << std::endl;
	for (int i = 0; i < input.size(); i++)
	{
		forward_propagation_each_node(i, temp_node);
		*old_node_file << temp_node[1][0] << " " << temp_node[1][1] << " " << std::endl;
	}
}
double mlp::error_cal(vector<double> cal_output, int data_num)
{
	double temp = 0;
	for (int i = 0; i < cal_output.size(); i++)
	{
		temp += (output[data_num][i] - cal_output[i]);
	}
	return pow(temp, 2) / 2;
}
void mlp::init_bias()
{
	this->bias = vector<vector<double>>(layer_num + 1, vector<double>(node_num, 0));
	for (int i = 0; i < layer_num; i++)
	{
		for (int j = 0; j < node_num; j++)
		{
			bias[i][j] = (rand() % 1000) / double(1000);
		}
	}
	bias[layer_num] = vector<double>(output_dim, 0);
	for (int i = 0; i < output_dim; i++)
	{
		bias[layer_num][i] = (rand() % 1000) / double(1000);
	}

	this->bias_weight = vector<vector<double>>(layer_num + 1, vector<double>(node_num, 0));
	for (int i = 0; i < layer_num; i++)
	{
		for (int j = 0; j < node_num; j++)
		{
			bias_weight[i][j] =0;
		}
	}
	bias_weight[layer_num] = vector<double>(output_dim, 0);
	for (int i = 0; i < output_dim; i++)
	{
		bias_weight[layer_num][i] = 0;
	}
}
void mlp::set_bias(vector<vector<double>> bias)
{
	this->bias = bias;
}
void mlp::random_init_weight()
{
	{
		vector<vector<double>> temp_layer;
		for (int j = 0; j < input_dim; j++)
		{
			vector<double> node_temp_layer;
			for (int z = 0; z < node_num; z++)
			{
				node_temp_layer.push_back(((rand() % 1000)) / double(1000));
			}
			temp_layer.push_back(node_temp_layer);
		}
		old_weight[0] = temp_layer;
	}
	for (int i = 1; i < layer_num; i++)
	{
		for (int j = 0; j < node_num; j++)
		{
			for (int z = 0; z < node_num; z++)
			{
				old_weight[i][j][z] = ((rand() % 1000)) / double(1000);
			}
		}
	}
	{
		vector<vector<double>> temp_layer;
		for (int j = 0; j < node_num; j++)
		{
			vector<double> node_temp_layer;
			for (int z = 0; z < output_dim; z++)
			{
				node_temp_layer.push_back(((rand() %1000) / double(1000)));
			}
			temp_layer.push_back(node_temp_layer);
		}
		old_weight[layer_num] = temp_layer;
	}
}
double mlp::activate_function(double net)
{
	/*return 1 / (1 + exp(-net));*/
	if (net >= 0) return net;
	else return 0;
}
double mlp::activate_function_drv(double net)
{
	return (1 - activate_function(net))*activate_function(net);
}
void mlp::forward_propagation(int data_num)
{
	node[0] = input[data_num];
	for (int layer = 1; layer < layer_num + 2; layer++)
	{
		int first_current_node_num = old_weight[layer - 1].size();//���� layer�� dim ������ �����ϴ� ����.
		int second_current_node_num = old_weight[layer - 1][0].size(); //���� layer�� dim ������ �����ϴ� ����.
		vector<double> temp(second_current_node_num, 0);
		for (int dim = 0; dim < first_current_node_num; dim++)
		{
			temp = matrix::add(temp, matrix::scalar(old_weight[layer - 1][dim], node[layer - 1][dim]));
		}//update�ϰ��� �ϴ� layer���� �ϳ� ���� ������ �� dim���� weight����� output����ŭ ������� ��, temp�� �����ؼ� ���ؼ� ����.
		temp = matrix::add(temp, matrix::scalar(matrix::mul(bias[layer - 1], bias_weight[layer - 1]), -1));
		for (int i = 0; i < second_current_node_num; i++)
		{
			temp[i] = activate_function(temp[i]);
		}//temp ���� activate_function�� ����.
		node[layer] = temp;//temp�� �ش� layer�� output���� ���.
	}
}
void mlp::forward_propagation_each_node(int data_num, vector<vector<double>> &temp_node)
{
	temp_node = node;
	temp_node[0] = input[data_num];
	for (int layer = 1; layer < layer_num + 2; layer++)
	{
		int first_current_node_num = old_weight[layer - 1].size();
		int second_current_node_num = old_weight[layer - 1][0].size();
		vector<double> temp(second_current_node_num, 0);
		for (int dim = 0; dim < first_current_node_num; dim++)
		{
			temp = matrix::add(temp, matrix::scalar(old_weight[layer - 1][dim], temp_node[layer - 1][dim]));
		}
		temp = matrix::add(temp, matrix::scalar(matrix::mul(bias[layer - 1], bias_weight[layer - 1]), -1));
		for (int i = 0; i < second_current_node_num; i++)
		{
			temp[i] = activate_function(temp[i]);
		}
		temp_node[layer] = temp;
	}
}
void mlp::set_back_delta(int data_num)
{
	//vector<double> temp;
	//for (int i = 0; i < output_dim; i++)
	//{
	//	temp.push_back((-1)*(output[data_num][i] - node[layer_num + 1][i])*(1 - node[layer_num + 1][i])*(node[layer_num + 1][i]));
	//}//E�� �̺��� ���� activate_function(net)�� �̺��� ���� ���� ������ delta�� ���� 
	//delta[layer_num] = temp;
	//for (int layer = layer_num - 1; layer > -1; layer--)
	//{
	//	for (int i = 0; i < node_num; i++)
	//	{
	//		delta[layer][i] = matrix::dot(delta[layer + 1], old_weight[layer + 1][i]) * (1 - node[layer + 1][i])*(node[layer + 1][i]);
	//	}//���� layer�� delta���� weight���� ������ ���� ��� ���� �ڿ�, activate_function(net)���� �̺а��� ���Ͽ��� delta�� ���
	//}

	vector<double> temp;
	for (int i = 0; i < output_dim; i++)
	{
		if(node[layer_num+1][i] > 0.)
		temp.push_back((-1)*(output[data_num][i] - node[layer_num + 1][i])*1);
		else
		temp.push_back((-1)*(output[data_num][i] - node[layer_num + 1][i]) * (1/2));
	}//E�� �̺��� ���� activate_function(net)�� �̺��� ���� ���� ������ delta�� ���� 
	delta[layer_num] = temp;
	for (int layer = layer_num - 1; layer > -1; layer--)
	{
		for (int i = 0; i < node_num; i++)
		{
			if (node[layer + 1][i] > 0)
			delta[layer][i] = matrix::dot(delta[layer + 1], old_weight[layer + 1][i]) * 1;
			else
			delta[layer][i] = matrix::dot(delta[layer + 1], old_weight[layer + 1][i]) * 1/2;
		}//���� layer�� delta���� weight���� ������ ���� ��� ���� �ڿ�, activate_function(net)���� �̺а��� ���Ͽ��� delta�� ���
	}
}
void mlp::back_propagation()
{

	for (int layer = layer_num; layer > 0; layer--)
	{
		for (int i = 0; i < node_num; i++)
		{
			new_weight[layer][i] = matrix::add(old_weight[layer][i], matrix::scalar(delta[layer], node[layer][i] * (-1) * (learning_rate)));//delta���� output���� ���ؼ� ���� weight�� ������ learning rate�� ���ؼ�, �� ����
			//���� weight������ ��.
		}
		bias_weight[layer] = matrix::add(bias_weight[layer], matrix::scalar(matrix::mul(delta[layer], bias[layer]), (learning_rate)));//delta���� output���� ���ؼ� ���� bias-weight�� ������ learning rate�� ���ؼ�, �� ����
			//���� bias-weight������ ��.
	}
	for (int i = 0; i < input_dim; i++)//������ input dim�� hidden layer�� dim�� �ٸ��� ������ ���� ó��.
	{
		new_weight[0][i] = matrix::add(old_weight[0][i], matrix::scalar(delta[0], node[0][i] * (-1) * (learning_rate)));
	}
	bias_weight[0] = matrix::add(bias_weight[0], matrix::scalar(matrix::mul(delta[0], bias[0]), (learning_rate)));
}
void mlp::for_next()
{
	old_weight = new_weight;
}
void mlp::learning()
{
	int epoch = 0;//epoch ����
	bool correct = false;
	while (correct == false)//correct�� true�� �����Ǹ� �ݺ��� ����
	{
		correct = true;
		int incorrect_num = 0;//�� epoch���� Ʋ�� data set�� ������ ����.
		for (int i = 0; i < input.size(); i++)
		{
			forward_propagation(i);
			if (error_cal(node[layer_num + 1], i) >= tolerance)//foward propagation ��� tolerance���� error���� ���ٸ� backward propagation ����
			{
				incorrect_num++;
				correct = false;
				if (epoch % 1001 == 1000)//1000 epoch���� backward_propagation�ÿ� weight�� bias�� ��ȭ�� ����.
				{
					save_old_bias(epoch, i);
					save_old_node_file(epoch, i);
					save_old_weight(epoch, i);
				}
				set_back_delta(i);
				back_propagation();
				if (epoch % 1001 == 1000)
				{
					save_new_bias(epoch, i);
					save_node_file(epoch, i);
					save_new_weight(epoch, i);
				}
				for_next();
				if (epoch % 10000 == 0)//10000 epoch���� �н����¸� Ȯ���ϱ� ���� ������ ���
				{
					std::cout << "input data:";
					matrix::print_one_dim_matrix(input[i]);
					std::cout << "output data:";
					matrix::print_one_dim_matrix(output[i]);
					std::cout << "tolerance:" << tolerance << "error:";
					std::cout << error_cal(node[layer_num + 1], i) << std::endl;
					std::cout << "old_weight" << std::endl;
					matrix::print_three_dim_matrix(old_weight);
					std::cout << "bias" << std::endl;
					matrix::print_two_dim_matrix(bias);
					std::cout << "node" << std::endl;
					matrix::print_two_dim_matrix(node);
					std::cout << "delta" << std::endl;
					matrix::print_two_dim_matrix(delta);
					std::cout << "new_Wegiht" << std::endl;
					matrix::print_three_dim_matrix(new_weight);
					std::cout << "changed_bias_weight" << std::endl;
					matrix::print_two_dim_matrix(bias_weight);
					std::cout << "now epoch:" << epoch << std::endl << std::endl << std::endl;
				}
			}
		}
		epoch++;
		if (epoch % 100 == 0)//100 epoch���� error�� ����.
		{
			save_error_file(epoch, incorrect_num);
		}
		if (epoch > 100000)//30000epoch�� �Ѿ�� �н� ����
		{
			std::cout << "�н��� �����Ͽ����ϴ�.30000epoch �ʰ�" << std::endl;
			break;
		}
		if (correct == true)//�н� �����
		{
			std::cout << "epoch:" << epoch << std::endl << std::endl << std::endl;
			std::cout << std::endl << std::endl << "�н� ����" << std::endl;
			save_new_bias(epoch, 0);
			save_node_file(epoch, 0);
			save_new_weight(epoch, 0);
			std::cout << "�н��� weight��" << std::endl;
			matrix::print_three_dim_matrix(new_weight);
			std::cout << "�н��� bias-weight ��" << std::endl;
			matrix::print_two_dim_matrix(bias_weight);
			std::cout << "bias" << std::endl;
			matrix::print_two_dim_matrix(bias);
			std::cout << "�н��� ������ ��� ���\n";
			for (int i = 0; i < input.size(); i++)
			{
				std::cout << "input data:";
				matrix::print_one_dim_matrix(input[i]);
				std::cout << "output data:";
				matrix::print_one_dim_matrix(output[i]);
				forward_propagation(i);
				std::cout << "actual output:";
				matrix::print_one_dim_matrix(node[layer_num + 1]);
			}
		}
	}
}


