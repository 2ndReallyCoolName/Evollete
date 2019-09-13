#pragma once
#include <map>
#include <vector>
#include "math.h"

namespace evo {

	namespace nn {
		class activation{

			float alpha;

			std::map<const char*, std::vector<float>> errors;
			
			std::map<const char*, int> s2int;

			int str2int(const char* s) {
				return s2int[s];
			}

		public:
			activation(float alpha = 0.01);

			std::vector<float> linear(std::vector<float> v, const char* vect_name);
				
			std::vector<float> sigmoid(std::vector<float> v, const char* vect_name);

			std::vector<float> tanh(std::vector<float> v, const char* vect_name);

			std::vector<float> relu(std::vector<float> v, const char* vect_name);

			std::vector<float> leaky_relu(std::vector<float> v, const char* vect_name);

			std::vector<float> softplus(std::vector<float> v, const char* vect_name);

			std::vector<float> softmax(std::vector<float> v, const char* vect_name);

			std::vector<float> get_error(const char* vect_name) { return errors[vect_name]; }

			float get_error(const char* vect_name, int j) { return errors[vect_name]; }

			std::vector<float> activate(std::vector<float> v, const char* act_function, const char* vect_name = "a");


		};
	}
}