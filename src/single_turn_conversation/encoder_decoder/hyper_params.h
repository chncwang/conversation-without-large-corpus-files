#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>
#include <string>
#include "insnet/insnet.h"

using std::string;

enum Optimizer {
    ADAM = 0,
    ADAGRAD = 1,
    ADAMW = 2
};

struct HyperParams {
    int word_dim;
    int hidden_dim;
    float dropout;
    int batch_size;
    int beam_size;
    float learning_rate;
    float learning_rate_decay;
    float min_learning_rate;
    float warm_up_learning_rate;
    int warm_up_iterations;
    int word_cutoff;
    string word_file;
    bool word_finetune;
    float l2_reg;
    ::Optimizer optimizer;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(word_dim, hidden_dim, dropout, batch_size, learning_rate, warm_up_iterations,
                word_cutoff, word_finetune);
    }

    void print() const {
        std::cout << "word_dim:" << word_dim << std::endl
            << "hidden_dim:" << hidden_dim << std::endl
            << "dropout:" << dropout << std::endl
            << "batch_size:" << batch_size << std::endl
            << "beam_size:" << beam_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
            << "learning_rate_decay:" << learning_rate_decay << std::endl
            << "warm_up_learning_rate:" << warm_up_learning_rate << std::endl
            << "warm_up_iterations:" << warm_up_iterations << std::endl
            << "min_learning_rate:" << min_learning_rate << std::endl
	    << "word_cutoff:" << word_cutoff << std::endl
	    << "word_file:" << word_file << std::endl
    	    << "word_finetune:" << word_finetune << std::endl
    	    << "l2_reg:" << l2_reg << std::endl
    	    << "optimizer:" << optimizer << std::endl; 
    }
};

#endif
