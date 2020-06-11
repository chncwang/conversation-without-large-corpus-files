#ifndef SINGLE_TURN_CONVERSATION_PERPLEX_H
#define SINGLE_TURN_CONVERSATION_PERPLEX_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>

#include "N3LDG.h"

float computePerplex(const std::vector<Node *> &nodes, const std::vector<int> &answers, int len,
        float &hypo_ppl_log) {
    float log_sum = 0.0f;
    float hypo_ppl_log_sum = 0;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        dtype reciprocal_answer_prob = 1 / node.getVal().v[answers.at(i)];
        log_sum += log(reciprocal_answer_prob);
        int index = 0;
        for (int j = 0; j < node.getDim(); ++j) {
            if (node.getVal().v[j] >= node.getVal().v[answers.at(i)]) {
                ++index;
            }
        }
        hypo_ppl_log_sum += log(index);
    }

    hypo_ppl_log = hypo_ppl_log_sum;
    return log_sum;
}

#endif
