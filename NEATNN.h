#pragma once

#include <vector>
#include <unordered_map>
#include "RandomGenerator.h"

class NEATNN
{
public:
	NEATNN();
	NEATNN(int input_num, int output_num);
	NEATNN(const NEATNN& other);
	~NEATNN();

	NEATNN& operator=(const NEATNN& other);

	// Activation function enumeration
	enum Activation_function
	{
		RELU,
		TANH,
	};

	// Function that propagates the input and calculates the activation value of all nodes
	void execute(std::vector<float> input);

	// Returns the output of the NN given an activation function
	std::vector<float> getOutput(const Activation_function a_func);

	// Genetic operators to compute on the NN
	NEATNN crossOver(const NEATNN& parent2);
	void mutate();

	static float weight_mutation_probability;
	static float remove_connection_probability;
	static float add_connection_probability;
	static float add_node_probability;

private:

	struct Node;
	struct NodeConnection
	{
		NodeConnection(Node* in_node, Node* out_node, float weight);
		NodeConnection(Node* in_node, Node* out_node, float weight, long int innovation_num);

		float weight;
		Node* in;
		Node* out;
		int innovation_num;

		static long int global_innovation_number;

		bool operator==(const NodeConnection& other)
		{
			return in == other.in && out == other.out;
		}
	};
	struct Node
	{
		Node(short int index);
		Node(const Node& other); // Copy constructor (doesnt copy incoming connections list)

		short int index;
		float value;
		std::list<NodeConnection*> incoming_connections;
		float activation_sum;
		bool activated;
	};
	std::vector<Node*> nodes_list;
	std::unordered_map<short int, Node*> input_nodes_list, output_nodes_list; // Auxiliary map for fast access to input/output nodes
	std::vector<NodeConnection*> connections;

	int node_index; // Keeps track of the maximun index to give to a node


	Node* add_node();
	NodeConnection* add_connection(Node* in, Node* out, float weight);

	RandomGenerator rand_generator;

};

