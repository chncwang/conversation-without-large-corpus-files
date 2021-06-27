#ifndef SINGLE_TURN_CONVERSATION_PERPLEX_H
#define SINGLE_TURN_CONVERSATION_PERPLEX_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>

#include "insnet/insnet.h"

float computePerplex(const insnet::Node &node, int row, const std::vector<int> &answers,
        int &hit_count,
        std::vector<int> &hit_flags,
        int hit_beam) {
    if (hit_beam < 1) {
        std::cerr << "hit beam is less than 0" << hit_beam << std::endl;
        abort();
    }
    float log_sum = 0.0f;
    float count_sum = 0;
    hit_flags.clear();

    int col = node.size() / row;
    if (col * row != node.size()) {
        std::cerr << boost::format("computePerplex col:%1% node dim:%2%\n") % col % node.size()
            << std::endl;
    }
    for (int i = 0; i < col; ++i) {
        int answer = answers.at(i);
        if (answer < 0 || answer >= node.size()) {
            std::cerr << boost::format("answer:%1% dim:%2%") << answer << node.size() << std::endl;
            abort();
        }
#if USE_GPU
        const_cast<insnet::Node &>(node).val().copyFromDeviceToHost();
#endif
        float reciprocal_answer_prob = 1 / node.getVal()[row * i + answer];
        log_sum += log(reciprocal_answer_prob);

        bool hit = true;
        int larger_count = 0;
        for (int j = 0; j < row; ++j) {
            if (node.getVal()[row * i + j] >= node.getVal()[row * i + answer]) {
                if (++larger_count > hit_beam) {
                    hit = false;
                    break;
                }
            }
        }
        if (hit) {
            ++count_sum;
        }
        hit_flags.push_back(hit);
    }

    hit_count = count_sum;
    return log_sum;
}

#endif
