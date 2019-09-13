#include "activation.h"


Accel Ac("C:/Users/ninanpyo/source/repos/Evolette/Evolette/functions.kernel");


evo::nn::activation::activation(float alpha = 0.01): alpha(alpha) {
	m["a"];
	m["linear"] = linear;
	m["sigmoid"] = sigmoid;
	m["tanh"] = tanh;
	m["relu"] = relu;
	m["leaky_relu"] = leaky_relu;
	m["softplus"] = softplus;
	m["softmax"] = softmax;
}

std::vector<float> evo::nn::activation::linear(std::vector<float> v, const char* vect_name) {
	errors[vect_name] = std::vector<float>(v.size(), 1);
	return v;
}

std::vector<float> evo::nn::activation::sigmoid(std::vector<float> v, const char* vect_name) {
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "sigmoid");
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::tanh(std::vector<float> v, const char* vect_name) {
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "tanh");
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::relu(std::vector<float> v, const char* vect_name) {
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "relu");
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::leaky_relu(std::vector<float> v, const char* vect_name) {
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "leaky_relu", alpha);
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::softplus(std::vector<float> v, const char* vect_name) {
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "softplus");
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::softmax(std::vector<float> v, const char* vect_name) {
	float sm = evo::sum(Ac.call1v(v, "raise", 2.718281828459045));
	std::vector<std::vector<float>> res = Ac.callActivationFunction(v, "softmax", sm);
	errors[vect_name] = res[1];
	return res[0];
}

std::vector<float> evo::nn::activation::activate(std::vector<float> v, const char* act_function, const char* vect_name) {
	if ((act_function != NULL) && (act_function[0] == '\0')) {
		errors[vect_name] = Ac.vec(1, v.size());
		return v;
	}
	else {
		return m[act_function](v, vect_name);
	}
}