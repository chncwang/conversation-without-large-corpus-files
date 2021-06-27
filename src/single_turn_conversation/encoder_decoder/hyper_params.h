#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>
#include <string>
#include "insnet/insnet.h"

using std::string;
using namespace insnet;

enum Optimizer {
    ADAM = 0,
    ADAGRAD = 1,
    ADAMW = 2
};

struct HyperParams {
    int hidden_dim;
    int batch_size;
    dtype learning_rate;
    dtype dropout;
    dtype clip_grad;
    int warm_up_iterations;
    int word_cutoff;
    ::Optimizer optimizer;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(hidden_dim, batch_size, learning_rate, dropout, warm_up_iterations, word_cutoff,
                optimizer, clip_grad);
    }

    void print() const {
        std::cout << "hidden_dim:" << hidden_dim << std::endl
            << "batch_size:" << batch_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
            << "clip_grad:" << clip_grad << std::endl
            << "dropout:" << dropout << std::endl
            << "warm_up_iterations:" << warm_up_iterations << std::endl
	    << "word_cutoff:" << word_cutoff << std::endl
    	    << "optimizer:" << optimizer << std::endl; 
    }
};

#endif
