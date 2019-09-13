#include "Dense.h"


evo::nn:: Dense::Dense(int unit_size, const char* activation, std::vector<float> weight_params, std::vector<float> bias_params,
	bool bias_bool, bool store_vectors, const char* fp, float alpha = 0.1)
	:out_size(unit_size), in_size(0), activation(activation), weight_params(weight_params), bias_params(bias_params), bias_bool(bias_bool),
	store_vectors(store_vectors), timestamp(1), fp(fp)
{
	act = evo::nn::activation(alpha);
	if (fp == "") {
		initialize_layer();
	}
}

evo::nn::Dense::Dense(int unit_size, int input_size, const char* activation, std::vector<float> weight_params, std::vector<float> bias_params,
	bool bias_bool, bool store_vectors, const char* fp, float alpha = 0.1)
	:out_size(unit_size), in_size(input_size), activation(activation), weight_params(weight_params), bias_params(bias_params), bias_bool(bias_bool),
	store_vectors(store_vectors), timestamp(1), fp(fp)
{
	act = evo::nn::activation(alpha);
	if (fp == "") {
		initialize_layer();
	}
}

std::vector<float> evo::nn:: Dense::feedforward(std::vector<float> x) {
	vectors.insert(std::pair<const char*, std::vector<float>> ("x" + timestamp, x));
	std::vector<float> h = evo::matmul(x, weight);
	vectors.insert(std::pair<const char*, std::vector<float>>("h" + timestamp, h));
	timestamp++;
	return h;
}

void evo::nn::Dense::train(std::vector<float> loss_vector, float training_rate) {
	for (int i = timestamp - 1; timestamp > 0; timestamp--) {

		for (int j = 0; j < out_size; j++) {
			float l = loss_vector[j];

			for (int i = 0; i < in_size; i++) {
				gradient(l, i, j, training_rate);
			}
		}
	}
}

void evo::nn::Dense::gradient(float error, int i, int j, int timestamp, float training_rate) {
	float dy_dh = act.get_error("a" + timestamp)[j];
	const char* vc = "x" + timestamp;
	float dh_w = (vectors.find(vc)->second)[i];
	float dh_dx = weight[i][j];
	float cost_w = error * dy_dh * training_rate, cost_b = error* dy_dh* training_rate;
	cost_w *= dh_w;
	weight[i][j] += cost_w;
	bias[j] += cost_b;
}

void evo::nn::Dense::initialize_layer() {
	weight = evo::random_mtx(in_size, out_size, weight_params[0], weight_params[1]);
	bias = evo::random_vec(out_size, weight_params[0], weight_params[1]);
}