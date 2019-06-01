#include "RandomGenerator.h"

#include <ctime>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

boost::mt19937 RandomGenerator::global_generator(std::time(0));

RandomGenerator::RandomGenerator()
{
	boost::random::uniform_int_distribution<> dist(0, std::time(0));
	int seed = dist(global_generator);
	generator.seed(seed);
}


RandomGenerator::~RandomGenerator()
{
}

int RandomGenerator::randomInteger(int min, int max)
{
	boost::random::uniform_int_distribution<> dist(min, max);
	return dist(generator);
}

float RandomGenerator::randomFloat(float min, float max)
{
	boost::random::uniform_real_distribution<float> dist(min, max);
	return dist(generator);
}

float RandomGenerator::gaussianFloat(float mean, float std)
{
	boost::random::normal_distribution<float> dist(mean, std);

	return dist(generator);
}
