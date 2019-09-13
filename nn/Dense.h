#pragma once
#include "math.h"
#include "activation.h"

namespace evo{
	namespace nn {
		class Dense {
			mtx weight;
			std::vector<float> bias;
			int out_size, in_size, timestamp;
			const char* activation; const char* fp;
			std::vector<float> weight_params, bias_params;
			bool bias_bool, store_vectors;
			std::map<const char*, std::vector<float>> vectors;
			evo::nn::activation act;

			void gradient(float error, int i, int j, int timestamp, float training_rate = 0.6);

			void initialize_layer();

			void reset_timestamp() { timestamp = 1; }


		public:
			Dense(int unit_size, const char* activation = "softmax", std::vector<float> weight_params = { -1, 1 },
				std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, bool store_vectors = 1, const char* fp = "", float alpha=0.1);

			Dense(int unit_size, int input_size, const char* activation = "softmax", std::vector<float> weight_params = { -1, 1 },
				std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, bool store_vectors = 1, const char* fp = "", float alpha = 0.1);

			std::vector<float> feedforward(std::vector<float> x);

			void train(std::vector<float> loss_vector, float training_rate = 0.5);

			void modify_weight(int i, int j, int val) { weight[i][j] += val; }

			void modify_bias(int j, int val) { bias[j] += val; }

			int get_inputSize() const { return in_size; }

			int get_outputSize() const { return out_size; }

			const char* get_activation() const { return activation; }

			std::vector<float> get_vector(const char* name, int timestamp = 0) { return vectors[name]; }
		};
	}
}