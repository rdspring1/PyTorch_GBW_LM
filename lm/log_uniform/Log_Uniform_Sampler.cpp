#include "Log_Uniform_Sampler.h"

#include <unordered_set>
#include <cmath>
#include <stddef.h>
#include <thread>
#include <iostream>

Log_Uniform_Sampler::Log_Uniform_Sampler(const int range_max) : N(range_max), distribution(0.0, 1.0), prob(N, 0)
{
	for(int idx = 0; idx < N; ++idx)
	{
		prob[idx] = (log(idx+2) - log(idx+1)) / log(range_max+1);
	}
}

std::vector<float> Log_Uniform_Sampler::expected_count(const int num_tries, std::vector<long> samples)
{
	std::vector<float> freq;
	for(auto& idx : samples)
	{
		float value = -expm1(num_tries * log1p(-prob[idx]));
		freq.emplace_back(value);
	}
	return freq;
}

std::unordered_set<long> Log_Uniform_Sampler::sample(const size_t size, int* num_tries)
{
	std::unordered_set<long> data;
	const double log_N = log(N);

	while(data.size() != size)
	{
		*num_tries += 1;
		double x = distribution(generator);
		long value = (lround(exp(x * log_N)) - 1) % N;
		data.emplace(value);
	}
	return data;
}

std::unordered_set<long> Log_Uniform_Sampler::sample(const size_t size, std::unordered_set<long> labels)
{
	std::unordered_set<long> data;
	const double log_N = log(N);

	while(data.size() != size)
	{
		double x = distribution(generator);
		long value = (lround(exp(x * log_N)) - 1) % N;
		if( labels.find(value) == labels.end() )
		{
			data.emplace(value);
		}
	}
	return data;
}
