#include "activation.h"


Accel Ac("C:/Users/ninanpyo/source/repos/Evolette/Evolette/functions.kernel");


evo::nn::activation::activation(float alpha = 0.01): alpha(alpha) {
	s2int["linear"] = 0;
	s2int["sigmoid"] = 1;
	s2int["softplus"] = 2;
	s2int["softmax"] = 3;
	s2int["relu"] = 4;
	s2int["leaky_relu"] = 5;
	s2int["tanh"] = 6;
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
		switch (str2int(vect_name))
		{
		case 0:
			return linear(v, vect_name);
		case 1:
			return sigmoid(v, vect_name);
		case 2:
			return softplus(v, vect_name);
		case 3:
			return softmax(v, vect_name);
		case 4:
			return relu(v, vect_name);
		case 5:
			return leaky_relu(v, vect_name);
		case 6:
			return tanh(v, vect_name);
		}
	}
}